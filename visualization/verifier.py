from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    status: str
    correct: bool | None
    parsed_answer: str | None
    parsed_reference: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MathVerifier:
    """Centralized math-verify wrapper used by every rollout metric."""

    def __init__(self, timeout_seconds: int = 5) -> None:
        try:
            from math_verify import parse, verify
        except ImportError as error:
            raise ImportError("math-verify is required for answer scoring") from error
        self._parse = parse
        self._verify = verify
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _wrapped_reference(reference: str) -> str:
        return reference if "$" in reference else f"${reference}$"

    def __call__(self, generated_response: str, reference_answer: str) -> VerificationResult:
        try:
            parsed_reference = self._parse(
                self._wrapped_reference(reference_answer), fallback_mode="no_fallback"
            )
            if not parsed_reference:
                return VerificationResult(
                    "reference_parse_failure", None, None, None, "empty parsed reference"
                )
            parsed_answer = self._parse(generated_response, fallback_mode="no_fallback")
            if not parsed_answer:
                return VerificationResult(
                    "prediction_parse_failure",
                    None,
                    None,
                    repr(parsed_reference),
                    "empty parsed prediction",
                )
        except BaseException as error:
            return VerificationResult(
                "parser_failure", None, None, None, f"{type(error).__name__}: {error}"
            )

        try:
            correct = bool(
                self._verify(
                    parsed_reference,
                    parsed_answer,
                    timeout_seconds=self.timeout_seconds,
                    raise_on_error=True,
                )
            )
        except BaseException as error:
            return VerificationResult(
                "verification_error",
                None,
                repr(parsed_answer),
                repr(parsed_reference),
                f"{type(error).__name__}: {error}",
            )
        return VerificationResult(
            "correct" if correct else "incorrect",
            correct,
            repr(parsed_answer),
            repr(parsed_reference),
        )


def verification_statistics(records: list[dict[str, Any]]) -> dict[str, int]:
    stats = {
        "total_generations": len(records),
        "successfully_parsed": 0,
        "correct": 0,
        "incorrect": 0,
        "parser_failures": 0,
        "runtime_errors": 0,
    }
    for record in records:
        status = record.get("verification_status")
        if status in {"correct", "incorrect"}:
            stats["successfully_parsed"] += 1
            stats[status] += 1
        elif status in {"prediction_parse_failure", "reference_parse_failure", "parser_failure"}:
            stats["parser_failures"] += 1
        else:
            stats["runtime_errors"] += 1
    return stats
