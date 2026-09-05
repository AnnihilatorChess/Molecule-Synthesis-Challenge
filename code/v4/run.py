"""v4: candidate ensemble (union v1 + v2 beams) + round-trip cascade, then select-best.

Pools the neural candidate sets from all available retro models (vote-weighted), runs the
same lookup+round-trip+kNN cascade as v3, and finally selects the best submission across
ALL versions by held-out top-1 -> predictions/final/submission.csv.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import defaultdict
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
from common.select import select_best  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402

TOP_NEURAL = 10


def _merge(cand_lists: list) -> list:
    """Vote-weighted union of several per-row candidate lists -> one ranked list per row."""
    if not cand_lists:
        return None
    n = len(cand_lists[0])
    out = []
    for i in range(n):
        agg: dict[str, float] = defaultdict(float)
        for cands in cand_lists:
            if i < len(cands):
                for r, score in cands[i]:
                    agg[r] += math.exp(score)
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        out.append([(r, math.log(w)) for r, w in ranked])
    return out


def _load_all_neural(split: str):
    lists = []
    for v in ("v1", "v2"):
        p = paths.ARTIFACTS_DIR / v / "candidates.pkl"
        if p.exists():
            with open(p, "rb") as f:
                lists.append(pickle.load(f)[split])
    return lists


def _gather_rt(products, neural, lookup_table):
    srcs = set()
    for i, prod in enumerate(products):
        if neural is not None and i < len(neural):
            for r, _ in neural[i][:TOP_NEURAL]:
                srcs.add(r)
        if prod in lookup_table and len(lookup_table[prod]) > 1:
            srcs.update(lookup_table[prod].keys())
    return list(srcs)


def _assemble(products, neural, lookup_table, knn, rerank_fn, trust):
    nn_trim = [c[:TOP_NEURAL] for c in neural] if neural is not None else None
    raw = assemble.build_submission(products, lookup_table=lookup_table,
                                    knn_retriever=knn, neural_candidates=nn_trim,
                                    rerank_fn=rerank_fn, trust_unambiguous_lookup=trust)
    return safe_lines(raw, products)


def main() -> None:
    t0 = time.time()
    _, art, preds = paths.version_dirs("v4")
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)

    # forward model from v3 (optional)
    fwd_ck = paths.ARTIFACTS_DIR / "v3" / "fwd_ckpt_best.pt"
    fwd_model = load_model(fwd_ck)[0] if fwd_ck.exists() else None

    neural_h = _merge(_load_all_neural("holdout"))
    neural_t = _merge(_load_all_neural("test"))

    # ---- holdout ----
    holdout_prods = data.holdout_products_canon()
    holdout_tgt = read_lines(paths.HOLDOUT_TARGET)
    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    knn_h = KNNRetriever(data.train_pairs_canon())
    rerank_h = None
    if fwd_model is not None:
        rt = forward_predict(fwd_model, tok, _gather_rt(holdout_prods, neural_h, train_lookup))
        rerank_h = assemble.make_roundtrip_rerank(rt)

    best_trust, best_lines, best_score, scores = True, None, -1.0, {}
    for trust in (True, False):
        lines = _assemble(holdout_prods, neural_h, train_lookup, knn_h, rerank_h, trust)
        acc, _, _ = score_lists(lines, holdout_tgt)
        scores[f"trust_{trust}"] = acc
        if acc > best_score:
            best_trust, best_lines, best_score = trust, lines, acc
    write_submission(best_lines, preds / "v4_holdout_pred.csv", expected_n=len(holdout_prods))
    holdout_top1 = score_files(preds / "v4_holdout_pred.csv")
    print(f"[v4] holdout policies={scores} -> trust={best_trust} top1={holdout_top1:.4f}")

    # ---- test ----
    test_prods = data.load_test_canon()["canon"]
    full_lookup = load_lookup(paths.LOOKUP_PKL)
    knn_t = KNNRetriever([(p, r) for p, r in data.load_canon_pairs() if p and r])
    rerank_t = None
    if fwd_model is not None:
        rt = forward_predict(fwd_model, tok, _gather_rt(test_prods, neural_t, full_lookup))
        rerank_t = assemble.make_roundtrip_rerank(rt)
    test_lines = _assemble(test_prods, neural_t, full_lookup, knn_t, rerank_t, best_trust)
    write_submission(test_lines, preds / "v4_test_pred.csv", expected_n=len(test_prods))

    with open(art / "result.json", "w") as f:
        json.dump({"version": "v4", "holdout_policies": scores, "trust": best_trust,
                   "holdout_top1": holdout_top1, "minutes": round((time.time()-t0)/60, 1)}, f, indent=2)

    # ---- final selection across all versions ----
    select_best()
    print(f"[v4] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
