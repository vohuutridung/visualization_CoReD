from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl


@dataclass(frozen=True)
class WeightedSample:
    uuid: str
    problem: str
    answer: str
    steps: tuple[str, ...]
    step_records: tuple[dict[str, Any], ...]
    excluded_last_step_text: str
    outside_before_think: str
    outside_after_think: str


def group_weight_records(records: Iterable[dict[str, Any]]) -> dict[str, WeightedSample]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["uuid"]), []).append(record)
    output: dict[str, WeightedSample] = {}
    for uuid, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["step_index"]))
        indices = [int(item["step_index"]) for item in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(f"non-contiguous step indices for {uuid}: {indices}")
        first = ordered[0]
        output[uuid] = WeightedSample(
            uuid=uuid,
            problem=str(first["problem"]),
            answer=str(first["answer"]),
            steps=tuple(str(item["step_text"]) for item in ordered),
            step_records=tuple(ordered),
            excluded_last_step_text=str(first.get("excluded_last_step_text", "")),
            outside_before_think=str(first.get("outside_before_think", "")),
            outside_after_think=str(first.get("outside_after_think", "")),
        )
    return output


def load_weighted_samples(path: str | Path) -> dict[str, WeightedSample]:
    return group_weight_records(read_jsonl(path))
