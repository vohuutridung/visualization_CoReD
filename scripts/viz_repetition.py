#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.cache import JsonlCache
from visualization.config import load_config, runtime_value
from visualization.embedding import QwenStepEmbedder
from visualization.metrics.repetition import max_previous_bleu, max_previous_semantic
from visualization.provenance import save_run_config
from visualization.records import load_weighted_samples
from visualization.reporting import ExclusionReport


LOGGER = logging.getLogger("viz_repetition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute lexical and semantic step repetition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--embedding-model")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32, help="samples per embedding group")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def choose(value, fallback):
    return fallback if value is None else value


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = load_config(args.config)
    output_root = Path(
        runtime_value(args.output_dir, "CORED_OUTPUT_DIR", config["output_dir"])
    )
    run_name = config["run_name"]
    results_dir = output_root / "results" / run_name
    weights_path = Path(choose(args.weights, results_dir / "weights.jsonl"))
    samples = list(load_weighted_samples(weights_path).values())
    samples.sort(key=lambda sample: sample.uuid)
    if args.limit is not None:
        samples = samples[: args.limit]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    cache = JsonlCache(
        results_dir / "repetition.jsonl",
        key=lambda record: (str(record["uuid"]), int(record["step_index"])),
        resume=args.resume,
    )
    exclusions = ExclusionReport()
    embedding_model = runtime_value(
        args.embedding_model, "CORED_EMBEDDING_MODEL", config["embedding"]["name"]
    )
    for record in cache.values():
        if record.get("embedding_model") != embedding_model:
            raise ValueError(
                f"repetition cache uses a different embedding model: {cache.path}"
            )
    embedder = QwenStepEmbedder(
        embedding_model,
        revision=config["embedding"].get("revision"),
        device_map=args.device_map,
        dtype=config["backbone"].get("dtype", "bfloat16"),
        max_length=config["embedding"]["max_length"],
    )
    started = time.monotonic()
    completed = 0
    for group_start in range(0, len(samples), args.batch_size):
        group = []
        for sample in samples[group_start : group_start + args.batch_size]:
            expected = [(sample.uuid, index) for index in range(1, len(sample.steps))]
            if all(key in cache for key in expected):
                completed += 1
                continue
            if len(sample.steps) < 2:
                exclusions.add("no_predecessor_step", sample.uuid)
                continue
            group.append(sample)
        if not group:
            continue

        flat_steps = [step for sample in group for step in sample.steps]
        try:
            flat_embeddings = embedder.encode(
                flat_steps, batch_size=args.embedding_batch_size
            )
        except RuntimeError as error:
            reason = "oom" if "out of memory" in str(error).lower() else "embedding_failure"
            for sample in group:
                exclusions.add(reason, sample.uuid, f"{type(error).__name__}: {error}")
            continue
        cursor = 0
        for sample in group:
            count = len(sample.steps)
            embeddings = flat_embeddings[cursor : cursor + count]
            cursor += count
            try:
                bleu_values = max_previous_bleu(sample.steps)
                semantic_values = max_previous_semantic(embeddings)
                for index in range(1, count):
                    bleu = float(bleu_values[index])
                    semantic = float(semantic_values[index])
                    if not math.isfinite(bleu) or not math.isfinite(semantic):
                        raise FloatingPointError(f"non-finite repetition metric at step {index}")
                    weight = sample.step_records[index]
                    cache.append(
                        {
                            "uuid": sample.uuid,
                            "step_index": index,
                            "step_text": sample.steps[index],
                            "weight": weight["weight"],
                            "is_high_weight": weight["is_high_weight"],
                            "weight_group": weight["weight_group"],
                            "bleu_repetition": bleu,
                            "semantic_repetition": semantic,
                            "bleu_tokenizer": "nltk.wordpunct_tokenize",
                            "bleu_smoothing": "SmoothingFunction.method1",
                            "embedding_model": embedding_model,
                        }
                    )
            except Exception as error:
                exclusions.add("metric_failure", sample.uuid, f"{type(error).__name__}: {error}")
                continue
            completed += 1
        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "repetition samples=%d/%d steps=%d samples/sec=%.3f",
            completed,
            len(samples),
            len(cache.records),
            completed / elapsed,
        )

    exclusions.write(output_root / "reports" / f"{run_name}_repetition_exclusions.json")
    save_run_config(
        output_root / "configs" / f"{run_name}_repetition.json",
        {
            "embedding_model": embedding_model,
            "embedding_revision": config["embedding"].get("revision"),
            "embedding_mode": "symmetric, instruction-free, last-token pooled, L2 normalized",
            "embedding_batch_size": args.embedding_batch_size,
            "samples_requested": len(samples),
            "sample_uuids": [sample.uuid for sample in samples],
            "samples_completed": completed,
            "step_records": len(cache.records),
            "bleu": {
                "order": 4,
                "tokenizer": "nltk.wordpunct_tokenize",
                "smoothing": "SmoothingFunction.method1",
                "aggregation": "max over individual previous steps",
            },
            "first_step_excluded": True,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
