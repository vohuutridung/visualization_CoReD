from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run_config(path: str | Path, config: dict[str, Any]) -> None:
    payload = dict(config)
    payload["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload["git_commit"] = git_commit()
    atomic_write_json(path, payload)
