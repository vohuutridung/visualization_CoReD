#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.cache import JsonlCache
from visualization.config import load_config
from visualization.data import (
    TrajectoryParseError,
    deterministic_subset,
    load_analysis_dataset,
    load_subset,
    parse_longcot,
    save_subset,
)
from visualization.io import atomic_write_json, atomic_write_jsonl
from visualization.provenance import save_run_config
from visualization.reporting import ExclusionReport
from visualization.weights import (
    StandardizationStats,
    WeightParameters,
    extract_council_signals,
    finalize_weight_records,
    fit_standardization,
    load_council_model,
)


LOGGER = logging.getLogger("viz_extract_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen Phase-1 council step weights")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--council-checkpoint-dir")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--expert-pattern")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lambda-u", type=float)
    parser.add_argument("--lambda-d", type=float)
    parser.add_argument("--weight-floor", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--standardization-path")
    parser.add_argument(
        "--fit-standardization-on-analysis",
        action="store_true",
        help="Explicit methodological deviation: fit moments on the held-out analysis set",
    )
    parser.add_argument("--logit-chunk-size", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def choose(value, fallback):
    return fallback if value is None else value


def load_stats(path: str | Path) -> StandardizationStats:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    return StandardizationStats(
        mean_u=float(value["mean_u"]),
        std_u=float(value["std_u"]),
        mean_rho=float(value["mean_rho"]),
        std_rho=float(value["std_rho"]),
        source=str(value.get("source", path)),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = load_config(args.config)
    dataset_name = choose(args.dataset, config["dataset"]["name"])
    split = choose(args.split, config["dataset"]["split"])
    backbone = choose(args.backbone, config["backbone"]["name"])
    checkpoint_dir = choose(args.council_checkpoint_dir, config["council"].get("checkpoint_dir"))
    if checkpoint_dir is None:
        raise ValueError("--council-checkpoint-dir is required; checkpoint paths are never guessed")
    num_experts = choose(args.num_experts, config["council"]["num_experts"])
    pattern = choose(args.expert_pattern, config["council"]["expert_pattern"])
    output_root = Path(choose(args.output_dir, config["output_dir"]))
    seed = choose(args.seed, config["seed"])
    run_name = config["run_name"]
    results_dir = output_root / "results" / run_name
    reports_dir = output_root / "reports"
    subsets_dir = output_root / "subsets"
    results_dir.mkdir(parents=True, exist_ok=True)

    lambda_u = choose(args.lambda_u, config["weights"].get("lambda_u"))
    lambda_d = choose(args.lambda_d, config["weights"].get("lambda_d"))
    if lambda_u is None or lambda_d is None:
        raise ValueError(
            "Phase-2 lambda_u/lambda_d are unset in the paper and config; provide their actual "
            "training values with --lambda-u and --lambda-d"
        )
    parameters = WeightParameters(
        lambda_u=lambda_u,
        lambda_d=lambda_d,
        weight_floor=choose(args.weight_floor, config["weights"]["weight_floor"]),
        epsilon=choose(args.epsilon, config["weights"]["epsilon"]),
    )

    adapter_paths = [Path(checkpoint_dir) / pattern.format(index=index) for index in range(num_experts)]
    missing = [str(path) for path in adapter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase-1 expert checkpoints not found: {missing}")

    dataset = load_analysis_dataset(
        dataset_name,
        split,
        revision=config["dataset"].get("revision"),
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model, tokenizer, adapter_names = load_council_model(
        backbone,
        adapter_paths,
        revision=config["backbone"].get("revision"),
        dtype=config["backbone"].get("dtype", "bfloat16"),
        device_map=args.device_map,
    )
    raw_cache = JsonlCache(
        results_dir / "weights_raw.jsonl",
        key=lambda record: (str(record["uuid"]), int(record["step_index"])),
        resume=args.resume,
    )
    signature_payload = {
        "backbone": backbone,
        "revision": config["backbone"].get("revision"),
        "adapter_paths": [str(path.resolve()) for path in adapter_paths],
        "epsilon": parameters.epsilon,
        "council_enable_thinking": config["council"].get("enable_thinking"),
        "teacher_forcing_content_template": config["council"][
            "teacher_forcing_content_template"
        ],
        "parser": "first <think>, exact \\n\\n split, final non-empty step excluded",
    }
    extraction_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    for cached in raw_cache.values():
        if cached.get("extraction_signature") != extraction_signature:
            raise ValueError(
                f"weight cache provenance differs from this run: {raw_cache.path}; "
                "use a separate output directory"
            )
    exclusions = ExclusionReport()
    parsed_count = 0
    for row in dataset:
        uuid = str(row.get("uuid", "<missing>"))
        try:
            trajectory = parse_longcot(dict(row))
        except TrajectoryParseError as error:
            exclusions.add(error.reason, uuid, str(error))
            continue
        parsed_count += 1
        expected_keys = [(trajectory.uuid, index) for index in range(trajectory.eligible_num_steps)]
        if all(key in raw_cache for key in expected_keys):
            continue
        try:
            signals = extract_council_signals(
                model,
                tokenizer,
                adapter_names,
                trajectory,
                epsilon=parameters.epsilon,
                logit_chunk_size=choose(
                    args.logit_chunk_size, config["weights"]["logit_chunk_size"]
                ),
                enable_thinking=config["council"].get("enable_thinking"),
                content_template=config["council"]["teacher_forcing_content_template"],
            )
        except RuntimeError as error:
            reason = "oom" if "out of memory" in str(error).lower() else "weight_computation_failure"
            exclusions.add(reason, trajectory.uuid, f"{type(error).__name__}: {error}")
            continue
        except Exception as error:
            exclusions.add(
                "weight_computation_failure",
                trajectory.uuid,
                f"{type(error).__name__}: {error}",
            )
            continue

        for index, (step, signal) in enumerate(zip(trajectory.eligible_steps, signals)):
            raw_cache.append(
                {
                    "uuid": trajectory.uuid,
                    "problem": trajectory.problem,
                    "answer": trajectory.answer,
                    "step_index": index,
                    "step_text": step,
                    **signal,
                    "original_num_steps": trajectory.original_num_steps,
                    "eligible_num_steps": trajectory.eligible_num_steps,
                    "excluded_last_step_text": trajectory.excluded_last_step_text,
                    "outside_before_think": trajectory.outside_before_think,
                    "outside_after_think": trajectory.outside_after_think,
                    "extraction_signature": extraction_signature,
                }
            )
        if args.debug:
            LOGGER.debug(
                "uuid=%s steps=%r excluded=%r signals=%s",
                trajectory.uuid,
                trajectory.eligible_steps,
                trajectory.excluded_last_step_text,
                signals,
            )

    cached_records = list(raw_cache.values())
    cached_by_uuid: dict[str, list[dict]] = {}
    for record in cached_records:
        cached_by_uuid.setdefault(str(record["uuid"]), []).append(record)
    raw_records = []
    for uuid, records in cached_by_uuid.items():
        expected = int(records[0]["eligible_num_steps"])
        indices = sorted(int(record["step_index"]) for record in records)
        if len(records) != expected or indices != list(range(expected)):
            exclusions.add(
                "incomplete_weight_cache",
                uuid,
                f"expected={expected}, present_indices={indices}",
            )
            continue
        raw_records.extend(records)
    stats_path = choose(args.standardization_path, config["weights"].get("standardization_path"))
    if stats_path:
        stats = load_stats(stats_path)
    elif args.fit_standardization_on_analysis:
        stats = fit_standardization(
            raw_records,
            source="held-out analysis dataset (explicit methodological deviation)",
        )
    else:
        raise ValueError(
            "Phase-2 training-corpus standardization statistics are required. Supply "
            "--standardization-path, or explicitly opt into the non-default "
            "--fit-standardization-on-analysis deviation."
        )
    atomic_write_json(results_dir / "standardization.json", stats.__dict__)
    weighted = finalize_weight_records(raw_records, stats, parameters)
    atomic_write_jsonl(results_dir / "weights.jsonl", weighted)
    if args.debug:
        debug_by_uuid: dict[str, list[dict]] = {}
        for record in weighted:
            debug_by_uuid.setdefault(str(record["uuid"]), []).append(record)
        for uuid, records in debug_by_uuid.items():
            LOGGER.debug(
                "uuid=%s weights=%s high=%s low=%s",
                uuid,
                [record["weight"] for record in records],
                [record["step_index"] for record in records if record["is_high_weight"]],
                [record["step_index"] for record in records if not record["is_high_weight"]],
            )

    valid_uuids = sorted(
        {
            str(record["uuid"])
            for record in weighted
            if int(record["eligible_num_steps"]) >= 2
        }
    )
    too_short = {
        str(record["uuid"])
        for record in weighted
        if int(record["eligible_num_steps"]) < 2
    }
    for uuid in sorted(too_short):
        exclusions.add("fewer_than_two_eligible_steps", uuid)
    subset_specs = {
        "answer_gain_500.json": int(config["subsets"]["answer_gain"]),
        "expert_branching_1000.json": int(config["subsets"]["expert_branching"]),
    }
    for filename, requested in subset_specs.items():
        destination = subsets_dir / filename
        if destination.exists():
            load_subset(destination)
        else:
            selected = deterministic_subset(valid_uuids, requested, seed)
            save_subset(destination, selected, seed, requested)
            if len(selected) < requested:
                LOGGER.warning("%s: requested %d but selected %d", filename, requested, len(selected))

    exclusions.write(reports_dir / f"{run_name}_weights_exclusions.json")
    save_run_config(
        output_root / "configs" / f"{run_name}_weights.json",
        {
            "dataset": {
                "name": dataset_name,
                "split": split,
                "revision": config["dataset"].get("revision"),
                "fingerprint": getattr(dataset, "_fingerprint", None),
            },
            "backbone": backbone,
            "backbone_revision": config["backbone"].get("revision"),
            "council_checkpoint_paths": [str(path) for path in adapter_paths],
            "num_experts": num_experts,
            "seed": seed,
            "step_parser": {
                "source": "inside <think> only",
                "separator": "\\n\\n",
                "exclude_final_non_empty_step": True,
                "adaptive_truncation": False,
                "teacher_forcing_content_template": config["council"][
                    "teacher_forcing_content_template"
                ],
            },
            "weight_parameters": parameters.__dict__,
            "standardization": stats.__dict__,
            "parsed_samples": parsed_count,
            "valid_sample_uuids": valid_uuids,
            "raw_step_records": len(raw_records),
        },
    )
    LOGGER.info("wrote %d weighted steps from %d parsed samples", len(weighted), parsed_count)


if __name__ == "__main__":
    main()
