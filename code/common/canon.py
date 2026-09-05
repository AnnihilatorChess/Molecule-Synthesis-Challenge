"""Canonicalization helpers that mirror the official `top1_accuracy.py` exactly.

The official metric does, per line:
    set(MolToSmiles(MolFromSmiles(frag)) for frag in line.split('.') if MolFromSmiles(frag))
and compares the predicted set to the true set.  Everything here reproduces that so our
local scoring and submission assembly are byte-faithful to the grader.
"""
from __future__ import annotations

from multiprocessing import Pool

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon_smiles(smi: str) -> str | None:
    """Canonicalize a single molecule SMILES; None if RDKit can't parse it."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def canon_frag_list(reactant_string: str) -> list[str]:
    """Split a (possibly multi-fragment) reactant string on '.', canonicalize each
    fragment, drop invalid ones.  Mirrors `process_chemicals` in top1_accuracy.py."""
    out: list[str] = []
    for frag in reactant_string.split("."):
        frag = frag.strip()
        if not frag:
            continue
        c = canon_smiles(frag)
        if c is not None:
            out.append(c)
    return out


def canon_set(reactant_string: str) -> frozenset[str]:
    """The order-agnostic set the metric compares on."""
    return frozenset(canon_frag_list(reactant_string))


def canon_join(reactant_string: str) -> str | None:
    """Canonicalize a reactant string and re-join fragments in a deterministic
    (sorted) order. Returns None if nothing parses."""
    frags = sorted(set(canon_frag_list(reactant_string)))
    if not frags:
        return None
    return ".".join(frags)


def sets_equal(pred: str, true: str) -> bool:
    """Top-1 correctness for a single row, exactly as the grader computes it."""
    return canon_set(pred) == canon_set(true)


# ---- multiprocessing canonicalization for bulk product lists ----
def _cansmi(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def canon_many(smiles: list[str], njobs: int = 8) -> list[str | None]:
    if njobs <= 1 or len(smiles) < 2000:
        return [_cansmi(s) for s in smiles]
    with Pool(njobs) as pool:
        return pool.map(_cansmi, smiles)
