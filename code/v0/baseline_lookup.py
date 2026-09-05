"""v0 baseline: exact-match lookup (canonical product -> majority reactant set)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import data, paths  # noqa: E402
from common.lookup import best_reactants, build_lookup_from_pairs, load_lookup  # noqa: E402
from _baselib import write_and_score  # noqa: E402


def main() -> None:
    # holdout: lookup built from TRAIN split only (honest)
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    holdout_prods = data.holdout_products_canon()
    holdout_preds = [best_reactants(train_lookup, p) for p in holdout_prods]

    # test: lookup built from ALL 40k rows
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    test_prods = data.load_test_canon()["canon"]
    test_preds = [best_reactants(full_lookup, p) if p else None for p in test_prods]

    write_and_score("lookup", holdout_preds, holdout_prods, test_preds, test_prods)


if __name__ == "__main__":
    main()
