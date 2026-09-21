from __future__ import annotations

from itertools import combinations
from typing import Sequence

from .repetition import cosine, sentence_bleu4


def expert_pair_count(expert_count: int) -> int:
    if expert_count < 0:
        raise ValueError("expert count cannot be negative")
    return expert_count * (expert_count - 1) // 2


def semantic_branching(embeddings: Sequence[Sequence[float]]) -> float:
    if len(embeddings) < 2:
        raise ValueError("semantic branching requires at least two experts")
    distances = [1.0 - cosine(left, right) for left, right in combinations(embeddings, 2)]
    expected = expert_pair_count(len(embeddings))
    if len(distances) != expected:
        raise AssertionError("expert pair enumeration is inconsistent")
    return sum(distances) / expected


def mean_pairwise_bleu(continuations: Sequence[str]) -> float:
    if len(continuations) < 2:
        raise ValueError("pairwise BLEU requires at least two experts")
    values = [
        0.5 * (sentence_bleu4(left, right) + sentence_bleu4(right, left))
        for left, right in combinations(continuations, 2)
    ]
    return sum(values) / len(values)
