"""SMILES randomization for data augmentation (Tetko/Schwaller augmented transformer).

Generates a random atom-ordering SMILES of the same molecule, which teaches the encoder
that many strings denote the same molecule — the single biggest lever for seq2seq retro
top-1.  Multi-fragment strings ('A.B') are randomized as a whole.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def randomize_smiles(smi: str) -> str:
    """Return a random (non-canonical) SMILES of the same molecule; falls back to the
    input string if RDKit can't parse it."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    try:
        return Chem.MolToSmiles(m, doRandom=True, canonical=False)
    except Exception:
        return smi
