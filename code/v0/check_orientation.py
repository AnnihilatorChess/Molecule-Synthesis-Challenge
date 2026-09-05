"""v0: assert the data orientation before anything else.

We predict reactants for given products, i.e. the model maps RIGHT (products) -> LEFT
(reactants).  This hard-fails the run if a future data swap flips the columns.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import paths  # noqa: E402


def main() -> None:
    with open(paths.ARTIFACTS_DIR / "v0" / "canon_pairs.pkl", "rb") as f:
        canon_pairs = pickle.load(f)
    with open(paths.ARTIFACTS_DIR / "v0" / "test_products.pkl", "rb") as f:
        test = pickle.load(f)

    right_set = {p for p, r in canon_pairs if p}      # products (RIGHT)
    left_set = {r for p, r in canon_pairs if r}        # reactants (LEFT), whole-string canon
    test_canon = [s for s in test["canon"] if s]
    n = len(test_canon)

    frac_right = sum(1 for t in test_canon if t in right_set) / n
    frac_left = sum(1 for t in test_canon if t in left_set) / n
    print(f"test in train RIGHT (products):  {frac_right:.4f}")
    print(f"test in train LEFT  (reactants): {frac_left:.4f}")

    assert frac_right >= frac_left, (
        "Orientation check FAILED: test overlaps LEFT more than RIGHT — columns may be swapped."
    )
    assert frac_right > 0.10, (
        f"Orientation check FAILED: test/RIGHT overlap {frac_right:.3f} unexpectedly low."
    )
    print("orientation OK: predict reactants (LEFT) from products (RIGHT).")


if __name__ == "__main__":
    main()
