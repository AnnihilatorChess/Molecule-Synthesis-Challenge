"""v5: forward-likelihood reranking over all beam candidates.

Diagnostic showed the gold reactant set is in the v1∪v2 beam ~75.7% of the time but only
ranked #1 ~30% — the bottleneck is ranking, not coverage.  v5 reranks every pooled
candidate by a within-row blend of the retro beam score and the FORWARD model's
log P(product | candidate_reactants) (round-trip likelihood).  The blend weights and the
"trust unambiguous lookup" overlay are auto-tuned on the held-out split, then applied to test.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import data, paths  # noqa: E402
from common.canon import canon_set  # noqa: E402
from common.data import read_lines  # noqa: E402
from common.decode import forward_loglik, forward_predict, load_model  # noqa: E402
from common.finalize import safe_lines, write_submission  # noqa: E402
from common.knn import KNNRetriever  # noqa: E402
from common.local_metric import score_files, score_lists  # noqa: E402
from common.lookup import best_reactants, build_lookup_from_pairs, load_lookup  # noqa: E402
from common.select import select_best  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402

CAND_N = 20
WEIGHT_GRID = [(1, 0), (0, 1), (1, 0.5), (0.5, 1), (1, 1), (1, 2), (2, 1), (1, 3), (3, 1)]


def merge(lists):
    n = len(lists[0])
    out = []
    for i in range(n):
        agg = defaultdict(float)
        for L in lists:
            for r, s in L[i]:
                agg[r] += math.exp(s)
        out.append(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))
    return out


def load_neural(split):
    lists = []
    for v in ("v1", "v2"):
        p = paths.ARTIFACTS_DIR / v / "candidates.pkl"
        if p.exists():
            with open(p, "rb") as f:
                lists.append(pickle.load(f)[split])
    return merge(lists) if lists else None


def _z(xs):
    if len(xs) < 2:
        return [0.0] * len(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    if sd < 1e-9:
        return [0.0] * len(xs)
    return [(x - m) / sd for x in xs]


def compute_fwd(model, tok, products, pooled):
    pairs, index = [], []
    for i, row in enumerate(pooled):
        for j, (r, _) in enumerate(row):
            pairs.append((r, products[i]))  # forward: reactants -> product
            index.append((i, j))
    lls = forward_loglik(model, tok, pairs)
    fwd = [[0.0] * len(row) for row in pooled]
    for (i, j), ll in zip(index, lls):
        fwd[i][j] = ll
    return fwd


def rerank(pooled, fwd, a, b):
    preds = []
    for row, fl in zip(pooled, fwd):
        if not row:
            preds.append(None)
            continue
        zr = _z([sc for _, sc in row])
        zf = _z(fl)
        scores = [a * zr[k] + b * zf[k] for k in range(len(row))]
        best = max(range(len(row)), key=lambda k: scores[k])
        preds.append(row[best][0])
    return preds


def finalize(preds, products, lookup, knn, overlay):
    out = []
    for pred, prod in zip(preds, products):
        if overlay and prod in lookup and len(lookup[prod]) == 1:
            out.append(best_reactants(lookup, prod))
            continue
        if pred is None:
            pred = best_reactants(lookup, prod) if prod in lookup else knn.query(prod, 1)[0]
        out.append(pred)
    return out


def oracle_in_pool(pooled, gold):
    hit = sum(1 for row, g in zip(pooled, gold) if any(canon_set(r) == g for r, _ in row))
    return hit / len(gold)


def main():
    _, art, preds = paths.version_dirs("v5")
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)
    fwd_model, _ = load_model(paths.ARTIFACTS_DIR / "v3" / "fwd_ckpt_best.pt")

    # ---- holdout: tune ----
    products_h = data.holdout_products_canon()
    gold = [canon_set(t) for t in read_lines(paths.HOLDOUT_TARGET)]
    pooled_h = [row[:CAND_N] for row in load_neural("holdout")]
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    knn_h = KNNRetriever(data.train_pairs_canon())

    # forward signal quality: greedy forward on the (canonical) gold reactants
    react_can = [data.canon_smiles_multi(r) or r for r in read_lines(paths.HOLDOUT_TARGET)]
    fwd_chk = forward_predict(fwd_model, tok, react_can)
    fwd_top1 = sum(1 for rc, p in zip(react_can, products_h)
                   if fwd_chk.get(rc, "") == p) / len(products_h)
    print(f"[v5] forward model greedy top-1 (reactants->product) = {fwd_top1:.4f}")
    print(f"[v5] pool oracle (gold in top-{CAND_N}) = {oracle_in_pool(pooled_h, gold):.4f}")

    fwd_h = compute_fwd(fwd_model, tok, products_h, pooled_h)
    best = {"acc": -1.0}
    for a, b in WEIGHT_GRID:
        base = rerank(pooled_h, fwd_h, a, b)
        for overlay in (True, False):
            lines = safe_lines(finalize(base, products_h, train_lookup, knn_h, overlay), products_h)
            acc, _, _ = score_lists(lines, [t for t in read_lines(paths.HOLDOUT_TARGET)])
            if acc > best["acc"]:
                best = {"acc": acc, "a": a, "b": b, "overlay": overlay, "lines": lines}
    print(f"[v5] best holdout config: a={best['a']} b={best['b']} overlay={best['overlay']} "
          f"top1={best['acc']:.4f}")
    hpath = preds / "v5_holdout_pred.csv"
    write_submission(best["lines"], hpath, expected_n=len(products_h))
    holdout_top1 = score_files(hpath)

    # ---- test: apply best config ----
    products_t = data.load_test_canon()["canon"]
    pooled_t = [row[:CAND_N] for row in load_neural("test")]
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    knn_t = KNNRetriever([(p, r) for p, r in data.load_canon_pairs() if p and r])
    fwd_t = compute_fwd(fwd_model, tok, products_t, pooled_t)
    base_t = rerank(pooled_t, fwd_t, best["a"], best["b"])
    lines_t = safe_lines(finalize(base_t, products_t, full_lookup, knn_t, best["overlay"]), products_t)
    write_submission(lines_t, preds / "v5_test_pred.csv", expected_n=len(products_t))

    with open(art / "result.json", "w") as f:
        json.dump({"version": "v5", "a": best["a"], "b": best["b"], "overlay": best["overlay"],
                   "forward_top1": fwd_top1, "pool_oracle": oracle_in_pool(pooled_h, gold),
                   "holdout_top1": holdout_top1}, f, indent=2)
    print(f"[v5] holdout top-1 = {holdout_top1:.4f}")
    select_best()


if __name__ == "__main__":
    main()
