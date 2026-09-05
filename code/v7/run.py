"""v7: learned reranker over pooled ensemble candidates.

The ensemble pool has oracle 0.82 but top-1 only ~0.37 — pure ranking headroom.  v7 fits a
classifier P(candidate is correct | features) and ranks by it.  Features per candidate:
consensus retro score, ensemble vote-count, forward round-trip log-likelihood, lookup priors,
fragment count / length, and atom-balance (reactant heavy atoms vs product).

Honest evaluation: the retro models never trained on the 2k holdout, so its candidate features
are unbiased.  We report **5-fold out-of-fold** top-1 on the holdout (group = row), avoiding the
holdout-tuning optimism that inflated v5/v6.  For the test submission the reranker is refit on
the full holdout.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem, RDLogger  # noqa: E402

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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MODELS = ["v1", "v2", "v6big", "v6s1", "v6s2"]
CAND_PER_MODEL = 20
CAND_N = 25
FEATS = ["retro", "retro_z", "retro_rank", "votes", "votes_frac", "fwd", "fwd_z",
         "fwd_rank", "in_lookup", "is_lk_cand", "is_lk_maj", "lk_freq", "n_frag",
         "length", "atom_bal", "atom_bal_abs", "covers"]

_ha = {}


def heavy_atoms(s):
    if s in _ha:
        return _ha[s]
    n = 0
    for f in s.split("."):
        m = Chem.MolFromSmiles(f)
        if m:
            n += m.GetNumHeavyAtoms()
    _ha[s] = n
    return n


def present_models():
    return [v for v in MODELS if (paths.ARTIFACTS_DIR / v / "candidates.pkl").exists()]


def gather(split, models):
    lists = []
    for v in models:
        with open(paths.ARTIFACTS_DIR / v / "candidates.pkl", "rb") as f:
            lists.append(pickle.load(f)[split])
    n = len(lists[0])
    out = []
    for i in range(n):
        agg, votes = defaultdict(float), defaultdict(int)
        for L in lists:
            for r, s in L[i][:CAND_PER_MODEL]:
                agg[r] += math.exp(s)
            for r in {r for r, _ in L[i][:CAND_PER_MODEL]}:
                votes[r] += 1
        cands = sorted(agg, key=lambda r: agg[r], reverse=True)[:CAND_N]
        out.append([(r, math.log(agg[r]), votes[r]) for r in cands])
    return out


def _z(xs):
    if len(xs) < 2:
        return [0.0] * len(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [0.0] * len(xs) if sd < 1e-9 else [(x - m) / sd for x in xs]


def _ranks(xs):  # 0 = best (highest), normalized to [0,1]
    order = sorted(range(len(xs)), key=lambda k: xs[k], reverse=True)
    r = [0] * len(xs)
    for pos, k in enumerate(order):
        r[k] = pos / max(1, len(xs) - 1)
    return r


def build_features(pooled, fwd, products, lookup, n_models, gold=None):
    """Return X (np.array), groups (row idx), per-(row,cand) bookkeeping, and y if gold given."""
    X, groups, rows_cands, y = [], [], [], []
    for i, (row, fl, prod) in enumerate(zip(pooled, fwd, products)):
        if not row:
            rows_cands.append([])
            continue
        retro = [s for _, s, _ in row]
        votes = [float(v) for _, _, v in row]
        rz, rr = _z(retro), _ranks(retro)
        fz, fr = _z(fl), _ranks(fl)
        lk = lookup.get(prod)
        lk_total = sum(lk.values()) if lk else 0
        lk_maj = best_reactants(lookup, prod) if lk else None
        p_atoms = heavy_atoms(prod)
        idxs = []
        for j, (cand, _, _) in enumerate(row):
            ra = heavy_atoms(cand)
            feat = [
                retro[j], rz[j], rr[j], votes[j], votes[j] / n_models,
                fl[j], fz[j], fr[j],
                1.0 if lk else 0.0,
                1.0 if (lk and cand in lk) else 0.0,
                1.0 if cand == lk_maj else 0.0,
                (lk[cand] / lk_total) if (lk and cand in lk and lk_total) else 0.0,
                float(cand.count(".") + 1),
                float(len(cand)),
                float(ra - p_atoms), float(abs(ra - p_atoms)),
                1.0 if ra >= p_atoms else 0.0,
            ]
            X.append(feat)
            groups.append(i)
            idxs.append((j, cand))
            if gold is not None:
                y.append(1 if canon_set(cand) == gold[i] else 0)
        rows_cands.append(idxs)
    X = np.asarray(X, dtype=np.float64)
    return (X, np.asarray(groups), rows_cands, np.asarray(y) if gold is not None else None)


def pick_per_row(prob, rows_cands, pooled):
    """Argmax prob per row -> chosen candidate string (None if row had no candidates)."""
    flat = [(i, cand) for i, idxs in enumerate(rows_cands) for (_, cand) in idxs]
    assert len(flat) == len(prob), f"{len(flat)} != {len(prob)}"
    preds = [None] * len(pooled)
    best = {}
    for gi, (i, cand) in enumerate(flat):
        if i not in best or prob[gi] > best[i][0]:
            best[i] = (prob[gi], cand)
    for i, (_, cand) in best.items():
        preds[i] = cand
    return preds


def finalize(preds, products, lookup, knn, overlay):
    out = []
    for pred, prod in zip(preds, products):
        if overlay and prod in lookup and len(lookup[prod]) == 1:
            out.append(best_reactants(lookup, prod)); continue
        if pred is None:
            pred = best_reactants(lookup, prod) if prod in lookup else knn.query(prod, 1)[0]
        out.append(pred)
    return out


def make_clf(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                          max_depth=3, l2_regularization=1.0,
                                          min_samples_leaf=40, random_state=0)


def main():
    _, art, preds = paths.version_dirs("v7")
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)
    models = present_models()
    print(f"[v7] models: {models}")
    fwd_ck = paths.ARTIFACTS_DIR / "v6" / "fwd_ckpt_best.pt"
    if not fwd_ck.exists():
        fwd_ck = paths.ARTIFACTS_DIR / "v3" / "fwd_ckpt_best.pt"
    fwd_model, _ = load_model(fwd_ck)
    n_models = len(models)

    # ---- holdout features ----
    products_h = data.holdout_products_canon()
    tgt_lines = read_lines(paths.HOLDOUT_TARGET)
    gold = [canon_set(t) for t in tgt_lines]
    pooled_h = gather("holdout", models)
    oracle = sum(1 for row, g in zip(pooled_h, gold)
                 if any(canon_set(r) == g for r, _, _ in row)) / len(gold)
    print(f"[v7] pool oracle = {oracle:.4f}")

    def fwd_for(products, pooled):
        pairs, idx = [], []
        for i, row in enumerate(pooled):
            for j, (r, _, _) in enumerate(row):
                pairs.append((r, products[i])); idx.append((i, j))
        lls = forward_loglik(fwd_model, tok, pairs)
        fwd = [[0.0] * len(row) for row in pooled]
        for (i, j), ll in zip(idx, lls):
            fwd[i][j] = ll
        return fwd

    fwd_h = fwd_for(products_h, pooled_h)
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    knn_h = KNNRetriever(data.train_pairs_canon())
    Xh, gh, rc_h, yh = build_features(pooled_h, fwd_h, products_h, train_lookup, n_models, gold)

    # ---- 5-fold OOF model+overlay selection ----
    best = {"acc": -1.0}
    for kind in ("hgb", "logreg"):
        oof = np.zeros(len(Xh))
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(Xh, yh, gh):
            clf = make_clf(kind).fit(Xh[tr], yh[tr])
            oof[te] = clf.predict_proba(Xh[te])[:, 1]
        base = pick_per_row(oof, rc_h, pooled_h)
        for overlay in (True, False):
            lines = safe_lines(finalize(base, products_h, train_lookup, knn_h, overlay), products_h)
            acc, _, _ = score_lists(lines, tgt_lines)
            print(f"   {kind} overlay={overlay}: OOF top1={acc:.4f}")
            if acc > best["acc"]:
                best = {"acc": acc, "kind": kind, "overlay": overlay, "lines": lines}
    print(f"[v7] best honest OOF: {best['kind']} overlay={best['overlay']} top1={best['acc']:.4f}")
    hpath = preds / "v7_holdout_pred.csv"
    write_submission(best["lines"], hpath, expected_n=len(products_h))
    holdout_top1 = score_files(hpath)  # == OOF (honest)

    # ---- test: refit on FULL holdout, apply ----
    products_t = data.load_test_canon()["canon"]
    pooled_t = gather("test", models)
    fwd_t = fwd_for(products_t, pooled_t)
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    knn_t = KNNRetriever([(p, r) for p, r in data.load_canon_pairs() if p and r])
    Xt, gt, rc_t, _ = build_features(pooled_t, fwd_t, products_t, full_lookup, n_models)
    clf = make_clf(best["kind"]).fit(Xh, yh)
    prob_t = clf.predict_proba(Xt)[:, 1]
    base_t = pick_per_row(prob_t, rc_t, pooled_t)
    lines_t = safe_lines(finalize(base_t, products_t, full_lookup, knn_t, best["overlay"]), products_t)
    write_submission(lines_t, preds / "v7_test_pred.csv", expected_n=len(products_t))

    # feature importance (permutation-free: HGB has none; report logreg coefs if used)
    with open(art / "result.json", "w") as f:
        json.dump({"version": "v7", "models": models, "pool_oracle": oracle,
                   "clf": best["kind"], "overlay": best["overlay"],
                   "oof_holdout_top1": holdout_top1}, f, indent=2)
    print(f"[v7] honest holdout top-1 = {holdout_top1:.4f}")
    select_best()


if __name__ == "__main__":
    main()
