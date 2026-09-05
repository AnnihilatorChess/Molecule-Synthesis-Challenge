"""Exact-match lookup prior: canonical product -> most common reactant set.

Because ~21.6% of test products appear verbatim in train, a frequency-weighted
lookup table is a strong, free component of the cascade.  But the product->reactant
relation is one-to-many, so we store a Counter and expose the majority vote plus the
full distribution (so the reranker can choose among observed sets for ambiguous hits).
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

from .data import canon_reactions, load_reactions, Reaction


def build_lookup(rxns: list[Reaction]) -> dict[str, Counter]:
    """canonical_product -> Counter[reactants_joined_string]."""
    table: dict[str, Counter] = {}
    for product, reactants in canon_reactions(rxns):
        table.setdefault(product, Counter())[reactants] += 1
    return table


def build_lookup_from_pairs(pairs: list[tuple[str | None, str | None]]) -> dict[str, Counter]:
    """Build the lookup from already-canonicalized (product, reactants) pairs.
    Rows where either side is None are skipped."""
    table: dict[str, Counter] = {}
    for product, reactants in pairs:
        if product and reactants:
            table.setdefault(product, Counter())[reactants] += 1
    return table


def _tiebreak_key(item: tuple[str, int]):
    """most_common first; ties -> shortest joined string, then lexicographic."""
    reactants, count = item
    return (-count, len(reactants), reactants)


def best_reactants(table: dict[str, Counter], product_canon: str) -> str | None:
    counter = table.get(product_canon)
    if not counter:
        return None
    return min(counter.items(), key=_tiebreak_key)[0]


def is_ambiguous(table: dict[str, Counter], product_canon: str) -> bool:
    counter = table.get(product_canon)
    return bool(counter) and len(counter) > 1


def save_lookup(table: dict[str, Counter], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(table, f)


def load_lookup(path: Path) -> dict[str, Counter]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_from_file(train_path: Path | None = None) -> dict[str, Counter]:
    return build_lookup(load_reactions(train_path))
