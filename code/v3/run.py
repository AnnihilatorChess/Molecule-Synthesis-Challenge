"""v3: forward model + round-trip rerank + hybrid cascade.

Combines the best retro neural candidates (cached from v2, else v1) with the exact-match
lookup and a kNN fallback, and reranks ambiguous/uncovered rows by whether a forward
(reactants->product) model reproduces the query product.  Produces holdout + test
predictions; picks the lookup-trust policy that scores best on the held-out split.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import assemble, data, paths  # noqa: E402
from common.data import read_lines  # noqa: E402
from common.decode import forward_predict, load_model  # noqa: E402
from common.finalize import safe_lines, write_submission  # noqa: E402
from common.knn import KNNRetriever  # noqa: E402
from common.local_metric import score_files, score_lists  # noqa: E402
from common.lookup import build_lookup_from_pairs, load_lookup  # noqa: E402
from common.run_forward import run as train_forward  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402

TOP_NEURAL = 10  # neural candidates per row considered for round-trip rerank


def _load_neural():
    for v in ("v2", "v1"):
        p = paths.ARTIFACTS_DIR / v / "candidates.pkl"
        if p.exists():
            with open(p, "rb") as f:
                print(f"[v3] using neural candidates from {v}")
                return pickle.load(f)
    print("[v3] WARNING: no neural candidates found; cascade = lookup+kNN only")
    return None


def _gather_rt_sources(products, neural_cands, lookup_table):
    """Candidate reactant strings that the cascade might rerank -> need round-trip."""
    srcs = set()
    for i, prod in enumerate(products):
        if neural_cands is not None and i < len(neural_cands):
            for r, _ in neural_cands[i][:TOP_NEURAL]:
                srcs.add(r)
        if prod in lookup_table and len(lookup_table[prod]) > 1:
            srcs.update(lookup_table[prod].keys())
    return list(srcs)


def _assemble(products, neural_cands, lookup_table, knn_retr, rerank_fn, trust):
    neural = None
    if neural_cands is not None:
        neural = [c[:TOP_NEURAL] for c in neural_cands]
    raw = assemble.build_submission(
        products, lookup_table=lookup_table, knn_retriever=knn_retr,
        neural_candidates=neural, rerank_fn=rerank_fn,
        trust_unambiguous_lookup=trust)
    return safe_lines(raw, products)


def main() -> None:
    t0 = time.time()
    _, art, preds = paths.version_dirs("v3")
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)

    # forward model (train unless already present)
    fwd_ck = art / "fwd_ckpt_best.pt"
    if not fwd_ck.exists():
        train_forward(version="v3", augment=True, epochs=30)
    fwd_model, _ = load_model(fwd_ck)

    neural = _load_neural()
    neural_h = neural["holdout"] if neural else None
    neural_t = neural["test"] if neural else None

    # ---- holdout ----
    holdout_prods = data.holdout_products_canon()
    holdout_tgt = read_lines(paths.HOLDOUT_TARGET)
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    knn_h = KNNRetriever(data.train_pairs_canon())

    rt_src_h = _gather_rt_sources(holdout_prods, neural_h, train_lookup)
    rt_pred_h = forward_predict(fwd_model, tok, rt_src_h)
    rerank_h = assemble.make_roundtrip_rerank(rt_pred_h)

    scores = {}
    best_trust, best_lines, best_score = True, None, -1.0
    for trust in (True, False):
        lines = _assemble(holdout_prods, neural_h, train_lookup, knn_h, rerank_h, trust)
        acc, _, _ = score_lists(lines, holdout_tgt)
        scores[f"trust_{trust}"] = acc
        if acc > best_score:
            best_trust, best_lines, best_score = trust, lines, acc
    hpath = preds / "v3_holdout_pred.csv"
    write_submission(best_lines, hpath, expected_n=len(holdout_prods))
    holdout_top1 = score_files(hpath)
    print(f"[v3] holdout policies={scores} -> trust={best_trust} top1={holdout_top1:.4f}")

    # ---- test (full-data lookup + kNN; chosen policy) ----
    test_prods = data.load_test_canon()["canon"]
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    full_pairs = [(p, r) for p, r in data.load_canon_pairs() if p and r]
    knn_t = KNNRetriever(full_pairs)

    rt_src_t = _gather_rt_sources(test_prods, neural_t, full_lookup)
    rt_pred_t = forward_predict(fwd_model, tok, rt_src_t)
    rerank_t = assemble.make_roundtrip_rerank(rt_pred_t)
    test_lines = _assemble(test_prods, neural_t, full_lookup, knn_t, rerank_t, best_trust)
    write_submission(test_lines, preds / "v3_test_pred.csv", expected_n=len(test_prods))

    summary = {"version": "v3", "holdout_policies": scores, "trust": best_trust,
               "holdout_top1": holdout_top1, "minutes": round((time.time() - t0) / 60, 1)}
    with open(art / "result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[v3] done in {summary['minutes']} min; holdout top-1 = {holdout_top1:.4f}")


if __name__ == "__main__":
    main()
