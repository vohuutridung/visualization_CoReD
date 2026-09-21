from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to read experiment configurations") from error
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        elif value is not None:
            output[key] = value
    return output


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing required configuration value: {dotted_key}")
        value = value[part]
    if value is None:
        raise ValueError(
            f"configuration value {dotted_key} is intentionally unset; supply the actual "
            "Phase-2 experimental value rather than guessing"
        )
    return value
