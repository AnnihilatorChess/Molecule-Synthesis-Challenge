"""v0 baseline: identity (predict the product as its own single reactant). Sanity floor."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import data  # noqa: E402
from _baselib import write_and_score  # noqa: E402


def main() -> None:
    holdout_prods = data.holdout_products_canon()
    test_prods = data.load_test_canon()["canon"]
    # identity = the product itself; safe_lines will keep it (valid) verbatim
    write_and_score("identity", list(holdout_prods), holdout_prods,
                    list(test_prods), test_prods)


if __name__ == "__main__":
    main()
