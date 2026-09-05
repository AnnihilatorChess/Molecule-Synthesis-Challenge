"""v0: exploratory data analysis + build the shared tokenizer.

Writes artifacts/v0/eda_summary.json and artifacts/v0/tokenizer.json (shared vocab over
products + reactants + test products, used by encoder & decoder in v1+).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import data, paths  # noqa: E402
from common.tokenizer import SmilesTokenizer, tokenize  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def length_stats(strings):
    lens = [len(tokenize(s)) for s in strings]
    return {
        "mean": round(st.mean(lens), 2),
        "p50": pct(lens, 50),
        "p95": pct(lens, 95),
        "p99": pct(lens, 99),
        "max": max(lens),
    }


def main() -> None:
    pairs = data.load_canon_pairs()
    products = [p for p, r in pairs if p and r]
    reactants = [r for p, r in pairs if p and r]
    test = data.load_test_canon()
    test_canon = [s for s in test["canon"] if s]

    # n reactant fragments per reaction
    nfrag = Counter(len(r.split(".")) for r in reactants)
    n = len(reactants)
    nfrag_dist = {str(k): round(v / n, 4) for k, v in sorted(nfrag.items())}

    # multi-fragment fraction
    multi_frac = sum(v for k, v in nfrag.items() if k > 1) / n

    # near-identity: product is one of the reactant fragments
    near_identity = sum(1 for p, r in zip(products, reactants)
                        if p in set(r.split("."))) / n

    # product duplication / one-to-many
    prod_to_sets: dict[str, set] = {}
    for p, r in zip(products, reactants):
        prod_to_sets.setdefault(p, set()).add(r)
    n_unique = len(prod_to_sets)
    one_to_many = sum(1 for s in prod_to_sets.values() if len(s) > 1)

    # test overlap + ambiguity among overlaps
    overlap = [p for p in test_canon if p in prod_to_sets]
    ambiguous_overlap = sum(1 for p in overlap if len(prod_to_sets[p]) > 1)

    # shared tokenizer over everything
    tok = SmilesTokenizer.build_from_smiles(products + reactants + test_canon)
    tok.save(paths.TOKENIZER_JSON)
    n_unk = tok.n_unk(products + reactants + test_canon)

    summary = {
        "n_train_rows": len(pairs),
        "n_fully_parsed": n,
        "n_test": len(test["canon"]),
        "n_test_parsed": len(test_canon),
        "nfrag_distribution": nfrag_dist,
        "multi_fragment_fraction": round(multi_frac, 4),
        "near_identity_fraction": round(near_identity, 4),
        "n_unique_products": n_unique,
        "one_to_many_products": one_to_many,
        "one_to_many_fraction_of_unique": round(one_to_many / n_unique, 4),
        "test_train_product_overlap": len(overlap),
        "test_train_product_overlap_frac": round(len(overlap) / len(test_canon), 4),
        "ambiguous_overlap": ambiguous_overlap,
        "product_token_len": length_stats(products),
        "reactant_token_len": length_stats(reactants),
        "test_token_len": length_stats(test_canon),
        "vocab_size": tok.vocab_size,
        "n_unk_tokens": n_unk,
    }
    with open(paths.EDA_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
