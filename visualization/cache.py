from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

from .io import read_jsonl


class CacheProvenanceError(ValueError):
    """Existing cache content belongs to a different experimental definition."""


class JsonlCache:
    """Durable append-only JSONL with de-duplication for resumable inference."""

    def __init__(
        self,
        path: str | Path,
        key: Callable[[dict[str, Any]], Hashable],
        *,
        resume: bool = True,
    ) -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[Hashable, dict[str, Any]] = {}
        if resume:
            self._repair_tail()
            for record in read_jsonl(self.path):
                record_key = self.key(record)
                if record_key in self.records:
                    raise ValueError(f"duplicate cache key {record_key!r} in {self.path}")
                self.records[record_key] = record
        elif self.path.exists():
            raise FileExistsError(f"refusing to overwrite cache without --resume: {self.path}")

    def _repair_tail(self) -> None:
        """Remove only an incomplete final record left by an interrupted append."""

        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            cursor = end
            last_newline = -1
            while cursor > 0 and last_newline < 0:
                size = min(8192, cursor)
                cursor -= size
                handle.seek(cursor)
                block = handle.read(size)
                offset = block.rfind(b"\n")
                if offset >= 0:
                    last_newline = cursor + offset
            tail_start = last_newline + 1
            if tail_start == end:
                return
            handle.seek(tail_start)
            tail = handle.read(end - tail_start)
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError):
                handle.truncate(tail_start)
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __contains__(self, key: Hashable) -> bool:
        return key in self.records

    def get(self, key: Hashable) -> dict[str, Any] | None:
        return self.records.get(key)

    def append(self, record: dict[str, Any]) -> bool:
        record_key = self.key(record)
        if record_key in self.records:
            return False
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except ImportError:
                pass
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise OSError("short write while appending JSONL cache")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.records[record_key] = record
        return True

    def values(self) -> Iterable[dict[str, Any]]:
        return self.records.values()
