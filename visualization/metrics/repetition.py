from __future__ import annotations

import math
from typing import Sequence


def sentence_bleu4(candidate: str, reference: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.tokenize import wordpunct_tokenize

    candidate_tokens = wordpunct_tokenize(candidate)
    reference_tokens = wordpunct_tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def max_previous_bleu(steps: Sequence[str]) -> list[float | None]:
    output: list[float | None] = [None] if steps else []
    for index in range(1, len(steps)):
        output.append(max(sentence_bleu4(steps[index], steps[before]) for before in range(index)))
    return output


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions must match and be non-empty")
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cannot compute cosine for a zero embedding")
    return dot / (left_norm * right_norm)


def max_previous_semantic(embeddings: Sequence[Sequence[float]]) -> list[float | None]:
    output: list[float | None] = [None] if embeddings else []
    for index in range(1, len(embeddings)):
        output.append(max(cosine(embeddings[index], embeddings[before]) for before in range(index)))
    return output
