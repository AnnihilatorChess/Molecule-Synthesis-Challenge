"""v0 baseline: k-NN retrieval (nearest train product by ECFP4 Tanimoto -> its reactants)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import data  # noqa: E402
from common.knn import KNNRetriever  # noqa: E402
from _baselib import write_and_score  # noqa: E402

K = 1  # set >1 to enable similarity-weighted top-k vote


def main() -> None:
    t0 = time.time()
    train_pairs = data.train_pairs_canon()
    retr = KNNRetriever(train_pairs)
    print(f"built kNN index over {len(retr.fps)} train products in {time.time()-t0:.1f}s")

    holdout_prods = data.holdout_products_canon()
    holdout_preds = [retr.query(p, k=K)[0] if p else None for p in holdout_prods]

    # test: full-data index
    full_pairs = [(p, r) for p, r in data.load_canon_pairs() if p and r]
    retr_full = KNNRetriever(full_pairs)
    test_prods = data.load_test_canon()["canon"]
    t1 = time.time()
    test_preds = [retr_full.query(p, k=K)[0] if p else None for p in test_prods]
    print(f"kNN test queries done in {time.time()-t1:.1f}s")

    write_and_score("knn", holdout_preds, holdout_prods, test_preds, test_prods)


if __name__ == "__main__":
    main()
