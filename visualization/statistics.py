from __future__ import annotations

import math
import statistics as stdlib_statistics
from typing import Iterable, Sequence


def quantile(values: Sequence[float], probability: float) -> float:
    """R/NumPy type-7 linear quantile without a NumPy dependency."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q25": math.nan,
            "q75": math.nan,
        }
    if not all(math.isfinite(value) for value in data):
        raise ValueError("summary values must be finite")
    return {
        "count": len(data),
        "mean": stdlib_statistics.fmean(data),
        "median": stdlib_statistics.median(data),
        "std": stdlib_statistics.pstdev(data),
        "q25": quantile(data, 0.25),
        "q75": quantile(data, 0.75),
    }


def pooled_group_summary(
    records: Iterable[dict], value_key: str
) -> tuple[dict[str, dict], dict[str, list[float]]]:
    grouped = {"High-W": [], "Low-W": []}
    sample_ids = {"High-W": set(), "Low-W": set()}
    for record in records:
        group = str(record["weight_group"])
        if group not in grouped:
            raise ValueError(f"unexpected weight group: {group}")
        value = float(record[value_key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {value_key} for {record.get('uuid')}")
        grouped[group].append(value)
        sample_ids[group].add(str(record["uuid"]))
    summaries = {}
    for group in ("High-W", "Low-W"):
        summaries[group] = {
            "sample_count": len(sample_ids[group]),
            **describe(grouped[group]),
        }
    return summaries, grouped
