"""k-NN retrieval baseline / fallback.

Featurize train products with Morgan (ECFP4, 2048-bit) fingerprints; for a query
product return the reactants of the most similar train product (k=1), or a vote over
the top-k neighbours.  Used both as a standalone baseline and as a cascade fallback
(a kNN answer is always a real, valid train reactant set).
"""
from __future__ import annotations

from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

RADIUS = 2
NBITS = 2048


def _fp(product_canon: str):
    m = Chem.MolFromSmiles(product_canon)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=NBITS)


class KNNRetriever:
    def __init__(self, pairs: list[tuple[str, str]]):
        """pairs: list of (canonical_product, reactants_joined)."""
        self.fps = []
        self.reactants = []
        for product, reactants in pairs:
            fp = _fp(product)
            if fp is None:
                continue
            self.fps.append(fp)
            self.reactants.append(reactants)

    def query(self, product_canon: str, k: int = 1) -> tuple[str | None, float]:
        """Return (reactants, best_similarity). reactants=None if query unparseable
        or index empty."""
        if not self.fps:
            return None, 0.0
        q = _fp(product_canon)
        if q is None:
            return None, 0.0
        sims = DataStructs.BulkTanimotoSimilarity(q, self.fps)
        if k == 1:
            j = max(range(len(sims)), key=sims.__getitem__)
            return self.reactants[j], sims[j]
        # top-k similarity-weighted vote over reactant sets
        order = sorted(range(len(sims)), key=sims.__getitem__, reverse=True)[:k]
        votes: dict[str, float] = defaultdict(float)
        for j in order:
            votes[self.reactants[j]] += sims[j]
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        return best, sims[order[0]]
