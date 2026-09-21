#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.config import load_config, runtime_value
from visualization.io import atomic_write_json, read_jsonl
from visualization.plotting import make_figures
from visualization.statistics import pooled_group_summary


METRICS = {
    "answer_gain": {
        "file": "answer_gain.jsonl",
        "value": "answer_gain",
        "title": "Answer Gain",
        "ylabel": "Correctness change, $C_i-C_{i-1}$",
        "ylim": (-1.0, 1.0),
        "expected": "High > Low",
    },
    "removal_damage": {
        "file": "removal_damage.jsonl",
        "value": "removal_damage",
        "title": "Removal Damage",
        "ylabel": "Correctness loss, $C_{full}-C_{-i}$",
        "ylim": (-1.0, 1.0),
        "expected": "High > Low",
    },
    "bleu_repetition": {
        "file": "repetition.jsonl",
        "value": "bleu_repetition",
        "title": "BLEU Repetition",
        "ylabel": "Maximum previous-step BLEU-4",
        "ylim": (0.0, 1.0),
        "expected": "High < Low",
    },
    "semantic_repetition": {
        "file": "repetition.jsonl",
        "value": "semantic_repetition",
        "title": "Semantic Repetition",
        "ylabel": "Maximum previous-step cosine similarity",
        "ylim": (-1.0, 1.0),
        "expected": "High < Low",
    },
    "expert_branching": {
        "file": "expert_branching.jsonl",
        "value": "semantic_branching",
        "title": "Expert Branching",
        "ylabel": "Mean pairwise cosine distance",
        "ylim": (0.0, 2.0),
        "expected": "High > Low",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate saved metrics and make paper figures")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--figure-width", type=float, default=11.0)
    parser.add_argument("--figure-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(
        runtime_value(args.output_dir, "CORED_OUTPUT_DIR", config["output_dir"])
    )
    run_name = config["run_name"]
    results_dir = output_root / "results" / run_name
    tables_dir = output_root / "tables" / run_name
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metric_groups = {}
    table = []
    machine = {}
    for metric, spec in METRICS.items():
        path = results_dir / spec["file"]
        if not path.exists():
            continue
        summaries, groups = pooled_group_summary(read_jsonl(path), spec["value"])
        metric_groups[metric] = groups
        high = summaries["High-W"]
        low = summaries["Low-W"]
        difference = float(high["mean"]) - float(low["mean"])
        machine[metric] = {
            "High-W": high,
            "Low-W": low,
            "mean_difference_high_minus_low": difference,
            "expected_direction": spec["expected"],
            "hypothesis_supported_by_means": (
                difference > 0 if spec["expected"] == "High > Low" else difference < 0
            ),
        }
        table.append(
            {
                "metric": spec["title"],
                "high_w_mean": high["mean"],
                "low_w_mean": low["mean"],
                "high_minus_low": difference,
                "expected_direction": spec["expected"],
                "high_w_steps": high["count"],
                "low_w_steps": low["count"],
                "samples": len(
                    {
                        str(record["uuid"])
                        for record in read_jsonl(path)
                    }
                ),
            }
        )
    atomic_write_json(tables_dir / "metric_summary.json", machine)
    with (tables_dir / "metric_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["metric"])
        writer.writeheader()
        writer.writerows(table)
    make_figures(
        metric_groups,
        METRICS,
        figures_dir,
        run_name,
        figure_size=(args.figure_width, args.figure_height),
        dpi=args.dpi,
    )

    merged_exclusions = {}
    for path in sorted((output_root / "reports").glob(f"{run_name}_*_exclusions.json")):
        with path.open(encoding="utf-8") as handle:
            merged_exclusions[path.stem] = json.load(handle)
    atomic_write_json(output_root / "reports" / "exclusions.json", merged_exclusions)


if __name__ == "__main__":
    main()
