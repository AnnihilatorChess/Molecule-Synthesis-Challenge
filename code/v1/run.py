"""v1: seq2seq retrosynthesis baseline (no augmentation, beam-5)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.run_retro import run  # noqa: E402


def main() -> None:
    run(version="v1", augment=False, epochs=20, beam_width=5, tta_k=1)


if __name__ == "__main__":
    main()
