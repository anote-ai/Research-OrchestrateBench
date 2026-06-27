"""Shared CLI helpers."""

from __future__ import annotations

import shlex
from typing import Callable, Sequence


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_step(
    label: str,
    handler: Callable[[Sequence[str] | None], int | None],
    argv: Sequence[str],
) -> None:
    """Run another packaged CLI entrypoint and surface failures consistently."""
    print("$ " + label + " " + " ".join(shlex.quote(arg) for arg in argv))
    exit_code = handler(list(argv))
    if exit_code not in (None, 0):
        raise SystemExit(exit_code)
