from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import atomic_write_json


class ExclusionReport:
    def __init__(self, representative_limit: int = 5) -> None:
        self.representative_limit = representative_limit
        self.counts: dict[str, int] = defaultdict(int)
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(self, reason: str, uuid: str | None, detail: str | None = None) -> None:
        self.counts[reason] += 1
        if len(self.examples[reason]) < self.representative_limit:
            record: dict[str, Any] = {"uuid": uuid}
            if detail:
                record["detail"] = detail
            self.examples[reason].append(record)

    def payload(self) -> dict[str, Any]:
        return {
            "total_excluded_events": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "representative_examples": dict(sorted(self.examples.items())),
        }

    def write(self, path: str | Path) -> None:
        atomic_write_json(path, self.payload())
