"""v6 component: train an extra diverse augmented retro model (seed/size via args).

Usage: train_seed.py <version> <seed> [dim n_enc n_dec]
e.g.   train_seed.py v6s1 7   |   train_seed.py v6s2 99 384 6 6
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.run_retro import run  # noqa: E402


def main() -> None:
    version = sys.argv[1]
    seed = int(sys.argv[2])
    overrides = {}
    if len(sys.argv) >= 6:
        overrides = {"dim": int(sys.argv[3]), "n_heads": 8,
                     "n_enc": int(sys.argv[4]), "n_dec": int(sys.argv[5])}
    run(version=version, augment=True, epochs=50, beam_width=15, tta_k=5,
        eval_every=4, seed=seed, model_overrides=overrides)


if __name__ == "__main__":
    main()
