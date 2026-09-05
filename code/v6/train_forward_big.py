"""v6 component: a stronger forward model (reactants->product) for better round-trip reranking.

The v3 forward model (d=256, 30 epochs) only reached 0.19 greedy top-1, making round-trip a
weak signal. This trains d=384/6+6 for 80 epochs and saves to artifacts/v6/fwd_ckpt_best.pt
(picked up preferentially by v6/run.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.run_forward import run  # noqa: E402


def main() -> None:
    run(version="v6", augment=True, epochs=80, eval_every=4,
        model_overrides={"dim": 384, "n_heads": 8, "n_enc": 6, "n_dec": 6})


if __name__ == "__main__":
    main()
