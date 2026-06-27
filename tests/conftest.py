"""Test configuration shared across the repository.

Pytest runs directly from the repo root in local development and CI, so we add
the ``src`` directory to ``sys.path`` to make the package importable without
requiring an editable install first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
