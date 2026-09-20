"""Shared CLI behavior."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable


def run_cli(action: Callable[[], object]) -> None:
    try:
        result = action()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if result is not None:
        print(json.dumps(result, indent=2, default=str))
