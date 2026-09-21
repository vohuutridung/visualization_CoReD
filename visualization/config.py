from __future__ import annotations

import copy
import os
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


def runtime_value(cli_value: Any, environment_name: str, config_value: Any) -> Any:
    """Resolve runtime inputs as CLI > environment > configuration."""

    if cli_value is not None:
        value = cli_value
    else:
        environment_value = os.environ.get(environment_name)
        value = environment_value if environment_value not in (None, "") else config_value
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def resolve_expert_paths(
    *,
    cli_paths: list[str] | None,
    cli_checkpoint_dir: str | None,
    cli_pattern: str | None,
    council_config: dict[str, Any],
    num_experts: int,
) -> list[Path]:
    """Resolve arbitrary expert paths or a shared directory/pattern convention."""

    if cli_paths:
        values = cli_paths
    elif os.environ.get("CORED_EXPERT_PATHS"):
        values = os.environ["CORED_EXPERT_PATHS"].split(os.pathsep)
    elif council_config.get("expert_paths"):
        values = list(council_config["expert_paths"])
    else:
        checkpoint_dir = runtime_value(
            cli_checkpoint_dir,
            "CORED_COUNCIL_DIR",
            council_config.get("checkpoint_dir"),
        )
        if checkpoint_dir is None:
            raise ValueError(
                "Phase-1 experts are required. Supply repeated --expert-path values, "
                "CORED_EXPERT_PATHS, council.expert_paths, --council-checkpoint-dir, "
                "CORED_COUNCIL_DIR, or council.checkpoint_dir."
            )
        pattern = runtime_value(
            cli_pattern, "CORED_EXPERT_PATTERN", council_config.get("expert_pattern")
        )
        if pattern is None:
            raise ValueError(
                "an expert filename pattern is required when using a shared council directory"
            )
        values = [
            str(Path(checkpoint_dir) / pattern.format(index=index))
            for index in range(num_experts)
        ]
    paths = [Path(os.path.expanduser(os.path.expandvars(str(value)))) for value in values]
    if len(paths) != num_experts:
        raise ValueError(
            f"expected {num_experts} expert paths, received {len(paths)}: {paths}"
        )
    return paths
