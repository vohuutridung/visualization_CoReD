#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.cache import CacheProvenanceError, JsonlCache
from visualization.config import load_config
from visualization.data import load_subset
from visualization.embedding import QwenStepEmbedder
from visualization.generation import DecodingConfig, GenerationRequest, VLLMGenerator, rollout_seed
from visualization.metrics.branching import mean_pairwise_bleu, semantic_branching
from visualization.prompts import (
    audit_prompt_sources,
    branching_user_prompt,
    render_chat_prompt,
    require_clean_audit,
)
from visualization.provenance import save_run_config
from visualization.records import load_weighted_samples
from visualization.reporting import ExclusionReport


LOGGER = logging.getLogger("viz_expert_branching")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure deterministic Phase-1 expert branching")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--subset")
    parser.add_argument("--backbone")
    parser.add_argument("--council-checkpoint-dir")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--expert-pattern")
    parser.add_argument("--embedding-model")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
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
    samples_by_uuid = load_weighted_samples(
        choose(args.weights, results_dir / "weights.jsonl")
    )
    subset_path = Path(
        choose(args.subset, output_root / "subsets" / "expert_branching_1000.json")
    )
    uuids = load_subset(subset_path)
    if args.limit is not None:
        uuids = uuids[: args.limit]
    exclusions = ExclusionReport()
    samples = []
    for uuid in uuids:
        sample = samples_by_uuid.get(uuid)
        if sample is None:
            exclusions.add("subset_uuid_missing_weights", uuid)
        elif len(sample.steps) < 2:
            exclusions.add("fewer_than_two_eligible_steps", uuid)
        else:
            samples.append(sample)

    checkpoint_dir = choose(args.council_checkpoint_dir, config["council"].get("checkpoint_dir"))
    if checkpoint_dir is None:
        raise ValueError("--council-checkpoint-dir is required")
    num_experts = choose(args.num_experts, config["council"]["num_experts"])
    if num_experts < 2:
        raise ValueError("expert branching requires at least two Phase-1 experts")
    pattern = choose(args.expert_pattern, config["council"]["expert_pattern"])
    adapter_paths = [Path(checkpoint_dir) / pattern.format(index=index) for index in range(num_experts)]
    missing = [str(path) for path in adapter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase-1 expert checkpoints not found: {missing}")

    from transformers import AutoTokenizer

    backbone = choose(args.backbone, config["backbone"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        revision=config["backbone"].get("revision"),
        trust_remote_code=False,
    )
    generator = VLLMGenerator(
        backbone,
        revision=config["backbone"].get("revision"),
        tensor_parallel_size=choose(
            args.tensor_parallel_size, config["backbone"]["tensor_parallel_size"]
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=config["backbone"]["max_model_len"],
        enable_lora=True,
        max_loras=num_experts,
        max_lora_rank=config["council"]["max_lora_rank"],
        dtype=config["backbone"].get("dtype", "bfloat16"),
    )
    embedder = QwenStepEmbedder(
        choose(args.embedding_model, config["embedding"]["name"]),
        revision=config["embedding"].get("revision"),
        dtype=config["backbone"].get("dtype", "bfloat16"),
        max_length=config["embedding"]["max_length"],
    )
    decoding = DecodingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        max_new_tokens=choose(
            args.max_new_tokens, config["decoding"]["branching_max_new_tokens"]
        ),
    )
    generation_cache = JsonlCache(
        results_dir / "expert_branching_generations.jsonl",
        key=lambda record: (
            str(record["uuid"]), int(record["step_index"]), int(record["expert_index"])
        ),
        resume=args.resume,
    )
    for record in generation_cache.values():
        if record.get("decoding") != decoding.__dict__:
            raise ValueError(
                f"branching generation cache uses different decoding: {generation_cache.path}"
            )
    metric_cache = JsonlCache(
        results_dir / "expert_branching.jsonl",
        key=lambda record: (str(record["uuid"]), int(record["step_index"])),
        resume=args.resume,
    )
    embedding_model = choose(args.embedding_model, config["embedding"]["name"])
    for record in metric_cache.values():
        if record.get("embedding_model") != embedding_model:
            raise ValueError(
                f"branching metric cache uses a different embedding model: {metric_cache.path}"
            )
    prompt_cache = (
        JsonlCache(
            results_dir / "expert_branching_prompts.jsonl",
            key=lambda record: (str(record["uuid"]), int(record["step_index"])),
            resume=args.resume,
        )
        if args.save_prompts or args.debug
        else None
    )
    seed = choose(args.seed, config["seed"])
    started = time.monotonic()
    generated_now = 0
    tokens_now = 0
    for sample_number, sample in enumerate(samples, start=1):
        requests = []
        meta = {}
        view = SimpleNamespace(
            problem=sample.problem,
            answer=sample.answer,
            excluded_last_step_text=sample.excluded_last_step_text,
            outside_before_think=sample.outside_before_think,
            outside_after_think=sample.outside_after_think,
        )
        try:
            for step_index in range(len(sample.steps)):
                prefix = sample.steps[:step_index]
                user_prompt = branching_user_prompt(sample.problem, prefix)
                require_clean_audit(audit_prompt_sources(view, prefix, user_prompt))
                rendered = render_chat_prompt(
                    tokenizer,
                    user_prompt,
                    is_qwen3=bool(config["backbone"]["is_qwen3"]),
                    enable_thinking=False,
                )
                prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
                if prompt_cache is not None:
                    prompt_key = (sample.uuid, step_index)
                    existing_prompt = prompt_cache.get(prompt_key)
                    if (
                        existing_prompt is not None
                        and existing_prompt.get("prompt_sha256") != prompt_sha
                    ):
                        raise CacheProvenanceError(
                            f"cached branching debug prompt differs for {prompt_key}"
                        )
                    prompt_cache.append(
                        {
                            "uuid": sample.uuid,
                            "step_index": step_index,
                            "rendered_prompt": rendered,
                            "prompt_sha256": prompt_sha,
                        }
                    )
                shared_seed = rollout_seed(seed, "branching", sample.uuid, step_index)
                for expert_index, adapter_path in enumerate(adapter_paths):
                    key = (sample.uuid, step_index, expert_index)
                    if key in generation_cache:
                        cached = generation_cache.get(key)
                        if (
                            cached.get("prompt_sha256") != prompt_sha
                            or cached.get("expert_adapter") != str(adapter_path)
                        ):
                            raise CacheProvenanceError(
                                f"cached branching request differs for {key}"
                            )
                        continue
                    requests.append(
                        GenerationRequest(
                            request_id=key,
                            prompt=rendered,
                            seed=shared_seed,
                            lora_path=str(adapter_path),
                            lora_name=f"expert_{expert_index}",
                            stop=("\n\n",),
                        )
                    )
                    meta[key] = {
                        "uuid": sample.uuid,
                        "step_index": step_index,
                        "expert_index": expert_index,
                        "expert_adapter": str(adapter_path),
                        "prompt_sha256": prompt_sha,
                    }
        except CacheProvenanceError:
            raise
        except Exception as error:
            exclusions.add("prompt_construction_failure", sample.uuid, str(error))
            continue
        try:
            outputs = generator.generate(requests, decoding, batch_size=args.batch_size) if requests else []
        except Exception as error:
            reason = "oom" if "out of memory" in str(error).lower() else "generation_failure"
            exclusions.add(reason, sample.uuid, f"{type(error).__name__}: {error}")
            continue
        for output in outputs:
            key = tuple(output["request_id"])
            record = {**meta[key], **{name: value for name, value in output.items() if name != "request_id"}}
            record["decoding"] = decoding.__dict__
            generation_cache.append(record)
            generated_now += 1
            tokens_now += int(record["generated_token_count"])

        for step_index in range(len(sample.steps)):
            metric_key = (sample.uuid, step_index)
            if metric_key in metric_cache:
                continue
            records = [
                generation_cache.get((sample.uuid, step_index, expert_index))
                for expert_index in range(num_experts)
            ]
            if any(record is None for record in records):
                exclusions.add("incomplete_expert_continuations", sample.uuid, f"step={step_index}")
                continue
            continuations = [str(record["generated_text"]).strip() for record in records]
            if any(not value for value in continuations):
                exclusions.add("empty_expert_continuation", sample.uuid, f"step={step_index}")
                continue
            try:
                embeddings = embedder.encode(
                    continuations, batch_size=args.embedding_batch_size
                )
                branching = semantic_branching(embeddings)
                lexical = mean_pairwise_bleu(continuations)
                if not math.isfinite(branching) or not math.isfinite(lexical):
                    raise FloatingPointError("non-finite branching metric")
            except Exception as error:
                exclusions.add("branching_metric_failure", sample.uuid, str(error))
                continue
            weight = sample.step_records[step_index]
            metric_cache.append(
                {
                    "uuid": sample.uuid,
                    "step_index": step_index,
                    "step_text": sample.steps[step_index],
                    "weight": weight["weight"],
                    "is_high_weight": weight["is_high_weight"],
                    "weight_group": weight["weight_group"],
                    "expert_count": num_experts,
                    "semantic_branching": branching,
                    "mean_pairwise_bleu": lexical,
                    "continuation_finish_reasons": [record["finish_reason"] for record in records],
                    "continuation_terminations": [record["termination"] for record in records],
                    "terminated_by_delimiter": [
                        bool(record["terminated_by_delimiter"]) for record in records
                    ],
                    "embedding_model": embedding_model,
                }
            )
        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "branching samples=%d/%d steps=%d generations=%d generations/sec=%.3f",
            sample_number,
            len(samples),
            len(metric_cache.records),
            generated_now,
            generated_now / elapsed,
        )

    exclusions.write(output_root / "reports" / f"{run_name}_branching_exclusions.json")
    save_run_config(
        output_root / "configs" / f"{run_name}_expert_branching.json",
        {
            "backbone": backbone,
            "thinking_mode": False,
            "council_checkpoint_paths": [str(path) for path in adapter_paths],
            "num_experts": num_experts,
            "subset_path": str(subset_path),
            "sample_uuids": [sample.uuid for sample in samples],
            "decoding": decoding.__dict__,
            "stop": "\\n\\n",
            "embedding_model": embedding_model,
            "seed": seed,
            "generated_this_run": generated_now,
            "tokens_generated_this_run": tokens_now,
            "steps_completed": len(metric_cache.records),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
