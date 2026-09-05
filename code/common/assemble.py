"""Hybrid cascade that turns per-row candidate sources into one prediction per row.

Priority (per the plan):
  1. lookup hit & unambiguous  -> lookup majority (highest precision)
  2. lookup hit & ambiguous    -> rerank among {lookup observed sets, neural beam sets}
  3. no lookup hit             -> rerank among neural beam sets (best valid)
  4. no valid neural           -> kNN nearest reactant set
  5. last resort               -> None (finalize.safe_lines fills identity fallback)

`neural_candidates[i]` is a list of (reactants_joined_str, score) for row i, already
RDKit-valid and canonical, sorted by score descending (may be empty / the whole arg None).
`rerank_fn(product_canon, candidates) -> chosen_str` is the forward round-trip reranker;
if None, the highest-scoring / majority candidate is taken.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from .lookup import best_reactants, is_ambiguous


def make_roundtrip_rerank(rt_pred: dict[str, str]):
    """rerank_fn(product, candidates) -> first candidate whose forward-predicted product
    equals `product` (round-trip consistent); falls back to the top candidate."""
    def rerank(product: str, candidates: list[str]) -> str:
        for c in candidates:
            if rt_pred.get(c) == product:
                return c
        return candidates[0]
    return rerank


def _pick(product: str, candidates: list[str], rerank_fn):
    if not candidates:
        return None
    if rerank_fn is None or len(candidates) == 1:
        return candidates[0]
    return rerank_fn(product, candidates)


def build_submission(
    products_canon: list[str],
    lookup_table: dict[str, Counter] | None = None,
    knn_retriever=None,
    neural_candidates: list[list[tuple[str, float]]] | None = None,
    rerank_fn: Callable[[str, list[str]], str] | None = None,
    trust_unambiguous_lookup: bool = True,
) -> list[str | None]:
    out: list[str | None] = []
    for i, prod in enumerate(products_canon):
        neural = []
        if neural_candidates is not None and i < len(neural_candidates):
            neural = [r for r, _ in neural_candidates[i]]

        chosen: str | None = None
        in_lookup = bool(lookup_table) and prod in lookup_table
        if in_lookup:
            lk_best = best_reactants(lookup_table, prod)
            if trust_unambiguous_lookup and not is_ambiguous(lookup_table, prod):
                chosen = lk_best
            else:
                # ambiguous: rerank lookup-observed sets (freq order) + neural sets
                observed = [r for r, _ in lookup_table[prod].most_common()]
                cands = observed + [n for n in neural if n not in observed]
                chosen = _pick(prod, cands, rerank_fn) or lk_best
        else:
            chosen = _pick(prod, neural, rerank_fn)

        if chosen is None and knn_retriever is not None:
            chosen, _ = knn_retriever.query(prod, k=1)

        out.append(chosen)
    return out
