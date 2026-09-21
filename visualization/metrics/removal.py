from __future__ import annotations

from typing import Sequence


def removal_damages(full_correctness: float, ablated_correctness: Sequence[float]) -> list[float]:
    return [float(full_correctness) - float(value) for value in ablated_correctness]
