"""Top-level entry: select the best submission across all versions by held-out top-1."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.select import select_best  # noqa: E402

if __name__ == "__main__":
    select_best()
