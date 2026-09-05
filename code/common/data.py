"""Load and parse the Synthesis-Challenge data.

Train rows look like:  `reactants >> products`
  - LEFT  of '>>' = reactants  (what we PREDICT)
  - RIGHT of '>>' = products   (the model INPUT)

The test file contains products only (one per line); we must output reactants.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .canon import canon_join, canon_smiles


@dataclass
class Reaction:
    reactants_raw: str  # LEFT, raw string (may be multi-fragment, '.'-joined)
    product_raw: str    # RIGHT, raw string


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_reactions(path: Path | None = None) -> list[Reaction]:
    """Parse data_train.csv into (reactants, product) pairs."""
    path = path or paths.DATA_TRAIN
    rxns: list[Reaction] = []
    for ln in read_lines(path):
        if ">>" not in ln:
            continue
        left, right = ln.split(">>", 1)
        rxns.append(Reaction(left.strip(), right.strip()))
    return rxns


def load_test_products(path: Path | None = None) -> list[str]:
    path = path or paths.TEST_PRODUCTS
    return read_lines(path)


def canon_reactions(rxns: list[Reaction]) -> list[tuple[str, str]]:
    """Return [(canonical_product, canonical_reactants_joined), ...], dropping any
    row where either side fails to parse.  Reactants are fragment-sorted-joined."""
    out: list[tuple[str, str]] = []
    for r in rxns:
        p = canon_smiles_multi(r.product_raw)
        rr = canon_join(r.reactants_raw)
        if p is not None and rr is not None:
            out.append((p, rr))
    return out


def canon_smiles_multi(s: str) -> str | None:
    """Canonicalize a (possibly multi-fragment) product/reactant string, fragments
    sorted and '.'-joined. None if nothing parses."""
    return canon_join(s)


# ---- cached-artifact loaders (produced by v0/make_holdout.py) ----
def load_canon_pairs() -> list[tuple[str | None, str | None]]:
    with open(paths.ARTIFACTS_DIR / "v0" / "canon_pairs.pkl", "rb") as f:
        return pickle.load(f)


def load_test_canon() -> dict:
    """{"raw": [...], "canon": [...]} for the 10k test products."""
    with open(paths.ARTIFACTS_DIR / "v0" / "test_products.pkl", "rb") as f:
        return pickle.load(f)


def load_split() -> dict:
    """{"train": [idx...], "holdout": [idx...]} into the 40k train rows."""
    with open(paths.HOLDOUT_IDX) as f:
        return json.load(f)


def holdout_products_canon() -> list[str]:
    """Canonical product (RIGHT) for each held-out row, aligned to holdout_target.csv."""
    pairs = load_canon_pairs()
    split = load_split()
    return [pairs[i][0] or "" for i in split["holdout"]]


def train_pairs_canon() -> list[tuple[str, str]]:
    """Fully-parsed (product, reactants) canonical pairs for the TRAIN split only."""
    pairs = load_canon_pairs()
    split = load_split()
    out = []
    for i in split["train"]:
        p, r = pairs[i]
        if p and r:
            out.append((p, r))
    return out
