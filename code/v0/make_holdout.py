"""v0: build the fixed held-out split + cached canonical data + lookup table.

Outputs (all under artifacts/v0/):
  - canon_pairs.pkl        : list[(product_canon|None, reactants_canon|None)] aligned to the 40k train rows
  - test_products.pkl      : {"raw": [...], "canon": [...]} for the 10k test products
  - holdout_idx.json       : {"train": [...], "holdout": [...]} (indices into the 40k rows)
  - holdout_products.csv   : raw product (RIGHT) for held-out rows, one per line, no header
  - holdout_target.csv     : raw reactants (LEFT) for held-out rows, one per line, no header  <-- official --target
  - lookup.pkl             : FULL-data product_canon -> Counter[reactants_canon] (for test-time use)
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import paths  # noqa: E402
from common.canon import canon_join  # noqa: E402
from common.data import load_reactions, load_test_products  # noqa: E402
from common.lookup import build_lookup_from_pairs  # noqa: E402


def main() -> None:
    t0 = time.time()
    art = paths.ARTIFACTS_DIR / "v0"
    art.mkdir(parents=True, exist_ok=True)

    rxns = load_reactions()
    print(f"loaded {len(rxns)} reactions")

    # canonicalize all rows once
    canon_pairs: list[tuple[str | None, str | None]] = []
    for r in rxns:
        p = canon_join(r.product_raw)    # product = RIGHT (model input), fragment-sorted
        rr = canon_join(r.reactants_raw)  # reactants = LEFT (target), fragment-sorted
        canon_pairs.append((p, rr))
    n_ok = sum(1 for p, r in canon_pairs if p and r)
    print(f"canonicalized: {n_ok}/{len(canon_pairs)} rows fully parse")
    with open(art / "canon_pairs.pkl", "wb") as f:
        pickle.dump(canon_pairs, f)

    # test products
    test_raw = load_test_products()
    test_canon = [canon_join(s) for s in test_raw]
    n_test_ok = sum(1 for s in test_canon if s)
    print(f"test products: {len(test_raw)} ({n_test_ok} parse)")
    with open(art / "test_products.pkl", "wb") as f:
        pickle.dump({"raw": test_raw, "canon": test_canon}, f)

    # fixed split
    rng = random.Random(paths.SEED)
    idx = list(range(len(rxns)))
    rng.shuffle(idx)
    holdout = sorted(idx[: paths.N_HOLDOUT])
    train = sorted(idx[paths.N_HOLDOUT:])
    with open(paths.HOLDOUT_IDX, "w") as f:
        json.dump({"train": train, "holdout": holdout}, f)
    print(f"split: {len(train)} train / {len(holdout)} holdout")

    # holdout product/target files (RAW), one per line, no header
    with open(paths.HOLDOUT_PRODUCTS, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(rxns[i].product_raw for i in holdout) + "\n")
    with open(paths.HOLDOUT_TARGET, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(rxns[i].reactants_raw for i in holdout) + "\n")

    # FULL-data lookup table (for test-time predictions)
    table = build_lookup_from_pairs(canon_pairs)
    with open(paths.LOOKUP_PKL, "wb") as f:
        pickle.dump(table, f)
    print(f"lookup table: {len(table)} unique canonical products")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
