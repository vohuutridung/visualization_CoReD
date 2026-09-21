#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.config import load_config
from visualization.data import load_subset
from visualization.generation import DecodingConfig, VLLMGenerator
from visualization.provenance import save_run_config
from visualization.records import load_weighted_samples
from visualization.reporting import ExclusionReport
from visualization.rollout_metrics import run_answer_gain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefix Answer Gain with a frozen base model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--subset")
    parser.add_argument("--backbone")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-rollouts", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def choose(value, fallback):
    return fallback if value is None else value


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = load_config(args.config)
    output_root = Path(choose(args.output_dir, config["output_dir"]))
    run_name = config["run_name"]
    results_dir = output_root / "results" / run_name
    weights_path = Path(choose(args.weights, results_dir / "weights.jsonl"))
    subset_path = Path(choose(args.subset, output_root / "subsets" / "answer_gain_500.json"))
    all_samples = load_weighted_samples(weights_path)
    uuids = load_subset(subset_path)
    if args.limit is not None:
        uuids = uuids[: args.limit]
    exclusions = ExclusionReport()
    samples = []
    for uuid in uuids:
        sample = all_samples.get(uuid)
        if sample is None:
            exclusions.add("subset_uuid_missing_weights", uuid)
        elif len(sample.steps) < 2:
            exclusions.add("fewer_than_two_eligible_steps", uuid)
        else:
            samples.append(sample)

    from transformers import AutoTokenizer

    backbone = choose(args.backbone, config["backbone"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        revision=config["backbone"].get("revision"),
        trust_remote_code=False,
    )
    # Deliberately base-only: enable_lora is false and no adapter path is accepted.
    generator = VLLMGenerator(
        backbone,
        revision=config["backbone"].get("revision"),
        tensor_parallel_size=choose(
            args.tensor_parallel_size, config["backbone"]["tensor_parallel_size"]
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=config["backbone"]["max_model_len"],
        enable_lora=False,
        dtype=config["backbone"].get("dtype", "bfloat16"),
    )
    decoding = DecodingConfig(
        temperature=config["decoding"]["temperature"],
        top_p=config["decoding"]["top_p"],
        top_k=config["decoding"]["top_k"],
        min_p=config["decoding"]["min_p"],
        repetition_penalty=config["decoding"]["repetition_penalty"],
        max_new_tokens=choose(
            args.max_new_tokens, config["decoding"]["answer_gain_max_new_tokens"]
        ),
    )
    num_rollouts = choose(args.num_rollouts, config["decoding"]["num_rollouts"])
    seed = choose(args.seed, config["seed"])
    summary = run_answer_gain(
        samples,
        tokenizer=tokenizer,
        generator=generator,
        decoding=decoding,
        num_rollouts=num_rollouts,
        seed=seed,
        batch_size=args.batch_size,
        results_dir=results_dir,
        is_qwen3=bool(config["backbone"]["is_qwen3"]),
        resume=args.resume,
        save_prompts=args.save_prompts or args.debug,
        exclusions=exclusions,
    )
    exclusions.write(output_root / "reports" / f"{run_name}_answer_gain_exclusions.json")
    save_run_config(
        output_root / "configs" / f"{run_name}_answer_gain.json",
        {
            "backbone": backbone,
            "adapters": None,
            "thinking_mode": False,
            "subset_path": str(subset_path),
            "sample_uuids": [sample.uuid for sample in samples],
            "decoding": decoding.__dict__,
            "num_rollouts": num_rollouts,
            "seed": seed,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
