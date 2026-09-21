from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


STEP_SEPARATOR = "\n\n"
REQUIRED_FIELDS = ("problem", "answer", "longcot", "uuid")


class TrajectoryParseError(ValueError):
    """A categorized, reportable long-CoT parsing failure."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ParsedTrajectory:
    uuid: str
    problem: str
    answer: str
    longcot: str
    think_content: str
    eligible_steps: tuple[str, ...]
    excluded_last_step_text: str
    original_num_steps: int
    outside_before_think: str
    outside_after_think: str

    @property
    def eligible_num_steps(self) -> int:
        return len(self.eligible_steps)

    def debug_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["eligible_num_steps"] = self.eligible_num_steps
        return value


def _require_text(row: dict[str, Any], key: str) -> str:
    if key not in row:
        raise TrajectoryParseError("missing_field", f"missing required field {key!r}")
    value = row[key]
    if value is None:
        raise TrajectoryParseError("missing_field", f"field {key!r} is null")
    return str(value)


def parse_longcot(row: dict[str, Any]) -> ParsedTrajectory:
    """Parse the first strict ``<think>...</think>`` region.

    Splitting is exactly on two newline characters. Empty chunks are removed
    before the final non-empty answer step is excluded.
    """

    longcot = _require_text(row, "longcot")
    open_at = longcot.find("<think>")
    if open_at < 0:
        raise TrajectoryParseError("missing_think_open", "longcot has no <think> tag")
    content_at = open_at + len("<think>")
    close_at = longcot.find("</think>", content_at)
    if close_at < 0:
        raise TrajectoryParseError("missing_think_close", "longcot has no </think> tag")

    think_content = longcot[content_at:close_at].strip()
    if not think_content:
        raise TrajectoryParseError("empty_think", "the <think> region is empty")

    steps = tuple(part.strip() for part in think_content.split(STEP_SEPARATOR) if part.strip())
    if not steps:
        raise TrajectoryParseError("empty_think", "the <think> region has no non-empty step")
    eligible = steps[:-1]
    if not eligible:
        raise TrajectoryParseError(
            "no_eligible_step", "only the final answer step remains after parsing"
        )

    return ParsedTrajectory(
        uuid=_require_text(row, "uuid"),
        problem=_require_text(row, "problem"),
        answer=_require_text(row, "answer"),
        longcot=longcot,
        think_content=think_content,
        eligible_steps=eligible,
        excluded_last_step_text=steps[-1],
        original_num_steps=len(steps),
        outside_before_think=longcot[:open_at],
        outside_after_think=longcot[close_at + len("</think>") :],
    )


def load_analysis_dataset(
    name: str,
    split: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(name, split=split, revision=revision, cache_dir=cache_dir)
    missing = sorted(set(REQUIRED_FIELDS) - set(dataset.column_names))
    if missing:
        raise ValueError(f"dataset is missing required fields: {missing}")
    return dataset.select_columns(list(REQUIRED_FIELDS))


def iter_parsed(dataset: Iterable[dict[str, Any]]) -> Iterator[ParsedTrajectory]:
    for row in dataset:
        yield parse_longcot(dict(row))


def deterministic_subset(
    uuids: Iterable[str], size: int, seed: int = 42
) -> list[str]:
    """Sample UUIDs reproducibly, independently of incoming dataset order."""

    unique = sorted(set(str(value) for value in uuids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[: min(size, len(unique))]


def subset_digest(uuids: Sequence[str]) -> str:
    encoded = json.dumps(list(uuids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_subset(path: str | Path, uuids: Sequence[str], seed: int, requested: int) -> None:
    from .io import atomic_write_json

    atomic_write_json(
        path,
        {
            "seed": seed,
            "requested": requested,
            "selected": len(uuids),
            "dataset_shortfall": len(uuids) < requested,
            "sha256": subset_digest(uuids),
            "uuids": list(uuids),
        },
    )


def load_subset(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    uuids = [str(value) for value in payload["uuids"]]
    if payload.get("sha256") and payload["sha256"] != subset_digest(uuids):
        raise ValueError(f"subset checksum mismatch: {path}")
    return uuids
