"""Small Phase-0 config loader.

The checked-in .yaml files intentionally use JSON syntax, which is valid YAML.
This lets the scaffold stay dependency-light while preserving stable YAML paths.
Phase 1 can add PyYAML without changing the CLI contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for an invalid or unsupported configuration."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Phase-0 configs must use JSON-compatible YAML syntax: {config_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return value


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
