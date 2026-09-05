"""Error analysis: is the 0.82(oracle)->0.37(top1) gap recoverable signal or irreducible noise?

Runs on the 2k holdout (retro models never trained on it -> honest). Decomposes the gap and
characterises the mis-ranked-but-in-pool cases: where does gold rank, how multi-fragment-skewed
are failures, can the forward model discriminate, and (the key test) how often is our wrong pick
itself a *legitimate alternative* reaction for that product (an observed train alternative, or
forward-round-trip-consistent) — which would make the choice irreducibly ambiguous.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import data, paths  # noqa: E402
from common.canon import canon_set  # noqa: E402
from common.data import load_reactions, read_lines  # noqa: E402
from common.decode import forward_loglik, forward_predict, load_model  # noqa: E402
from common.lookup import build_lookup_from_pairs  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

# import v7 helpers
spec = importlib.util.spec_from_file_location("v7run", HERE / "v7" / "run.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


def hist(positions, bins=((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 9999))):
    n = len(positions)
    print(f"    (n={n})")
    for lo, hi in bins:
        c = sum(1 for p in positions if lo <= p <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")
        print(f"    rank {label:>5}: {c:5d} ({c/n:6.1%})")


def main():
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)
    fwd_ck = paths.ARTIFACTS_DIR / "v6" / "fwd_ckpt_best.pt"
    fwd_model, _ = load_model(fwd_ck)
    models = v7.present_models()

    products = data.holdout_products_canon()
    tgt = read_lines(paths.HOLDOUT_TARGET)
    gold = [canon_set(t) for t in tgt]
    pooled = v7.gather("holdout", models)

    # forward loglik per candidate
    pairs, idx = [], []
    for i, row in enumerate(pooled):
        for j, (r, _, _) in enumerate(row):
            pairs.append((r, products[i])); idx.append((i, j))
    lls = forward_loglik(fwd_model, tok, pairs)
    fwd = [[0.0] * len(row) for row in pooled]
    for (i, j), v in zip(idx, lls):
        fwd[i][j] = v

    train_lookup = build_lookup_from_pairs(data.train_pairs_canon())
    X, g, rc, y = v7.build_features(pooled, fwd, products, train_lookup, len(models), gold)

    # learned reranker OOF probabilities
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = v7.make_clf("logreg").fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    # map oof back to per-row candidate order
    prob = [[0.0] * len(row) for row in pooled]
    for pos, (i, j) in enumerate([(i, j) for i, ic in enumerate(rc) for (j, _) in ic]):
        prob[i][j] = oof[pos]

    N = len(products)
    not_in_pool = in_correct = in_wrong = 0
    gold_ranks = []          # rank of gold among reranked candidates (in-pool rows)
    wrong_gold_ranks = []    # same but only when our #1 != gold
    nfrag_total = Counter(); nfrag_correct = Counter()
    # forward discrimination
    gold_is_fwd_max = 0; in_pool_for_fwd = 0
    # ambiguity probes
    amb_rows = 0; pick_is_observed_alt = 0; gold_ne_majority = 0; wrong_n = 0

    for i, row in enumerate(pooled):
        g_set = gold[i]
        nf = len(g_set) if g_set else 1
        nfrag_total[nf] += 1
        if not row:
            not_in_pool += 1
            continue
        sets = [canon_set(r) for r, _, _ in row]
        order = sorted(range(len(row)), key=lambda k: prob[i][k], reverse=True)
        ranked_sets = [sets[k] for k in order]
        top_set = ranked_sets[0]
        in_gold = g_set in sets
        if not in_gold:
            not_in_pool += 1
            continue
        # gold rank (1-based) in reranked order
        gr = ranked_sets.index(g_set) + 1
        gold_ranks.append(gr)
        # forward discrimination
        in_pool_for_fwd += 1
        if max(range(len(row)), key=lambda k: fwd[i][k]) == sets.index(g_set):
            gold_is_fwd_max += 1
        if top_set == g_set:
            in_correct += 1
            nfrag_correct[nf] += 1
        else:
            in_wrong += 1
            wrong_gold_ranks.append(gr)
            wrong_n += 1
            # ambiguity: product observed in train with >1 reactant set?
            lk = train_lookup.get(products[i])
            if lk and len(lk) > 1:
                amb_rows += 1
                observed = {canon_set(s) for s in lk}
                if top_set in observed:
                    pick_is_observed_alt += 1
                from common.lookup import best_reactants
                if canon_set(best_reactants(train_lookup, products[i])) != g_set:
                    gold_ne_majority += 1

    print("=" * 60)
    print(f"HOLDOUT gap decomposition (N={N})")
    print(f"  gold NOT in pool (coverage ceiling) : {not_in_pool/N:6.1%}")
    print(f"  gold in pool & we pick it (top-1)   : {in_correct/N:6.1%}")
    print(f"  gold in pool but MIS-RANKED         : {in_wrong/N:6.1%}")
    print(f"  -> pool oracle = {(in_correct+in_wrong)/N:.3f}, realized = {in_correct/N:.3f}")

    print("\nGold rank among reranked candidates (in-pool rows):")
    hist(gold_ranks)
    print("\nGold rank for MIS-RANKED rows only (how close did we get):")
    hist(wrong_gold_ranks)

    print("\nMulti-fragment breakdown (top-1 accuracy by #gold fragments):")
    for nf in sorted(nfrag_total):
        t = nfrag_total[nf]; c = nfrag_correct.get(nf, 0)
        print(f"    {nf} frag: {c/t:6.1%}  (n={t})")

    print("\nForward-model discrimination (in-pool rows):")
    print(f"    gold has the HIGHEST forward-loglik in its row: {gold_is_fwd_max/in_pool_for_fwd:6.1%}")

    print("\nAmbiguity probe (mis-ranked rows whose product is in train with >1 reactant set):")
    if amb_rows:
        print(f"    such rows: {amb_rows} of {wrong_n} mis-ranked")
        print(f"    our wrong pick IS an observed train alternative for that product: {pick_is_observed_alt/amb_rows:6.1%}")
        print(f"    gold differs from the train-majority reactants:                  {gold_ne_majority/amb_rows:6.1%}")
    else:
        print("    (none)")


if __name__ == "__main__":
    main()
