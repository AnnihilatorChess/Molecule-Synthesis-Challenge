"""v2: augmented seq2seq retrosynthesis (on-the-fly source randomization, beam-10, TTA x5)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.run_retro import run  # noqa: E402


def main() -> None:
    run(version="v2", augment=True, epochs=50, beam_width=10, tta_k=5)


if __name__ == "__main__":
    main()
