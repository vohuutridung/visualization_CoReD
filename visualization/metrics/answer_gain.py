from __future__ import annotations

from typing import Sequence


def answer_gains(correctness_by_prefix: Sequence[float]) -> list[float]:
    if len(correctness_by_prefix) < 2:
        return []
    return [
        float(correctness_by_prefix[index]) - float(correctness_by_prefix[index - 1])
        for index in range(1, len(correctness_by_prefix))
    ]
