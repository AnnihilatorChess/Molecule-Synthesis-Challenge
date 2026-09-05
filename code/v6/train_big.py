"""v6 component: a bigger, longer-trained augmented retro model (d=384, 6+6, beam-20).

Tests how much candidate-pool oracle + top-1 we gain from model capacity. Saved as a
standalone version dir so select_best and the v6 ensemble can pick it up.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.run_retro import run  # noqa: E402


def main() -> None:
    run(version="v6big", augment=True, epochs=80, beam_width=20, tta_k=5,
        eval_every=4,
        model_overrides={"dim": 384, "n_heads": 8, "n_enc": 6, "n_dec": 6})


if __name__ == "__main__":
    main()
