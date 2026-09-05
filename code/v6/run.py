"""v6: multi-model ensemble + reranking.

Pools beam candidates from ALL available retro models (v1, v2, v6big, extra seeds...),
computing per candidate: (1) consensus retro log-score (vote-weighted across models),
(2) ensemble vote-count (#models whose beam contains it), (3) forward-loglik round-trip.
Reranks by a within-row z-score blend whose weights + the lookup overlay are tuned on the
held-out split, then applied to test.  Finally selects the best submission across all versions.
"""
from __future__ import annotations

import itertools
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
from common.decode import forward_loglik, load_model  # noqa: E402
from common.finalize import safe_lines, write_submission  # noqa: E402
from common.knn import KNNRetriever  # noqa: E402
from common.local_metric import score_files, score_lists  # noqa: E402
from common.lookup import best_reactants, build_lookup_from_pairs, load_lookup  # noqa: E402
from common.select import select_best  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402

CANDIDATE_MODELS = ["v1", "v2", "v6big", "v6s1", "v6s2", "v6s3"]
CAND_PER_MODEL = 20
CAND_N = 25
W_GRID = [0, 0.5, 1, 2]  # weights tried for each of (retro, fwd, votes)


def present_models():
    return [v for v in CANDIDATE_MODELS
            if (paths.ARTIFACTS_DIR / v / "candidates.pkl").exists()]


def gather(split, models):
    lists = []
    for v in models:
        with open(paths.ARTIFACTS_DIR / v / "candidates.pkl", "rb") as f:
            lists.append(pickle.load(f)[split])
    n = len(lists[0])
    out = []
    for i in range(n):
        agg = defaultdict(float)
        votes = defaultdict(int)
        for L in lists:
            for r, s in L[i][:CAND_PER_MODEL]:
                agg[r] += math.exp(s)
            for r in {r for r, _ in L[i][:CAND_PER_MODEL]}:
                votes[r] += 1
        cands = sorted(agg, key=lambda r: agg[r], reverse=True)[:CAND_N]
        out.append([(r, math.log(agg[r]), float(votes[r])) for r in cands])
    return out


def _z(xs):
    if len(xs) < 2:
        return [0.0] * len(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [0.0] * len(xs) if sd < 1e-9 else [(x - m) / sd for x in xs]


def compute_fwd(model, tok, products, pooled):
    pairs, idx = [], []
    for i, row in enumerate(pooled):
        for j, (r, _, _) in enumerate(row):
            pairs.append((r, products[i]))
            idx.append((i, j))
    lls = forward_loglik(model, tok, pairs)
    fwd = [[0.0] * len(row) for row in pooled]
    for (i, j), ll in zip(idx, lls):
        fwd[i][j] = ll
    return fwd


def rerank(pooled, fwd, wr, wf, wv):
    preds = []
    for row, fl in zip(pooled, fwd):
        if not row:
            preds.append(None)
            continue
        zr = _z([s for _, s, _ in row])
        zf = _z(fl)
        zv = _z([v for _, _, v in row])
        k = max(range(len(row)),
                key=lambda j: wr * zr[j] + wf * zf[j] + wv * zv[j])
        preds.append(row[k][0])
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


def main():
    _, art, preds = paths.version_dirs("v6")
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)
    models = present_models()
    print(f"[v6] ensembling models: {models}")
    # prefer the stronger v6 forward model if it has been trained
    fwd_ck = paths.ARTIFACTS_DIR / "v6" / "fwd_ckpt_best.pt"
    if not fwd_ck.exists():
        fwd_ck = paths.ARTIFACTS_DIR / "v3" / "fwd_ckpt_best.pt"
    print(f"[v6] forward model: {fwd_ck.parent.name}/{fwd_ck.name}")
    fwd_model, _ = load_model(fwd_ck)

    # ---- holdout: tune weights + overlay ----
    products_h = data.holdout_products_canon()
    tgt_lines = read_lines(paths.HOLDOUT_TARGET)
    gold = [canon_set(t) for t in tgt_lines]
    pooled_h = gather("holdout", models)
    oracle = sum(1 for row, g in zip(pooled_h, gold)
                 if any(canon_set(r) == g for r, _, _ in row)) / len(gold)
    print(f"[v6] ensemble pool oracle (gold in top-{CAND_N}) = {oracle:.4f}")
    fwd_h = compute_fwd(fwd_model, tok, products_h, pooled_h)
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    knn_h = KNNRetriever(data.train_pairs_canon())

    best = {"acc": -1.0}
    for wr, wf, wv in itertools.product(W_GRID, repeat=3):
        if wr == wf == wv == 0:
            continue
        base = rerank(pooled_h, fwd_h, wr, wf, wv)
        for overlay in (True, False):
            lines = safe_lines(finalize(base, products_h, train_lookup, knn_h, overlay), products_h)
            acc, _, _ = score_lists(lines, tgt_lines)
            if acc > best["acc"]:
                best = {"acc": acc, "wr": wr, "wf": wf, "wv": wv,
                        "overlay": overlay, "lines": lines}
    print(f"[v6] best holdout: wr={best['wr']} wf={best['wf']} wv={best['wv']} "
          f"overlay={best['overlay']} top1={best['acc']:.4f}")
    hpath = preds / "v6_holdout_pred.csv"
    write_submission(best["lines"], hpath, expected_n=len(products_h))
    holdout_top1 = score_files(hpath)

    # ---- test ----
    products_t = data.load_test_canon()["canon"]
    pooled_t = gather("test", models)
    fwd_t = compute_fwd(fwd_model, tok, products_t, pooled_t)
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    knn_t = KNNRetriever([(p, r) for p, r in data.load_canon_pairs() if p and r])
    base_t = rerank(pooled_t, fwd_t, best["wr"], best["wf"], best["wv"])
    lines_t = safe_lines(finalize(base_t, products_t, full_lookup, knn_t, best["overlay"]), products_t)
    write_submission(lines_t, preds / "v6_test_pred.csv", expected_n=len(products_t))

    with open(art / "result.json", "w") as f:
        json.dump({"version": "v6", "models": models, "pool_oracle": oracle,
                   "weights": [best["wr"], best["wf"], best["wv"]],
                   "overlay": best["overlay"], "holdout_top1": holdout_top1}, f, indent=2)
    print(f"[v6] holdout top-1 = {holdout_top1:.4f}")
    select_best()


if __name__ == "__main__":
    main()
