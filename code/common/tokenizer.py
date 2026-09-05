"""Regex SMILES tokenizer + simple vocab (reused from the Generation Challenge).

Pattern lifted from Schwaller et al. / SmilesPE. Handles:
- bracketed atoms: [nH], [O-], [C@@H], [NH3+], [C-], ...
- 2-letter atoms: Cl, Br, Si, Se
- ring digits %NN (e.g. %12)
- standard SMILES single chars, including '.' (fragment separator)

Special tokens prepended: <pad>=0, <bos>=1, <eos>=2, <unk>=3.

For retrosynthesis we build ONE shared vocab over reactants + products + test
products, used by both the encoder (source) and decoder (target).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# canonical SMILES regex (Schwaller 2018)
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|@@|@|%\d\d|\d|=|#|-|\+|\\|/|\.|\(|\)|"
    r"[A-Za-z])"
)

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize(smi: str) -> list[str]:
    return SMILES_REGEX.findall(smi)


class SmilesTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]
        self.vocab_size = len(self.token_to_id)

    # ---- factory methods ----
    @classmethod
    def build_from_smiles(cls, smiles: list[str], min_freq: int = 1) -> "SmilesTokenizer":
        counter: Counter = Counter()
        for s in smiles:
            counter.update(tokenize(s))
        kept = [t for t, c in counter.most_common() if c >= min_freq]
        # specials first to keep their ids stable
        vocab = list(SPECIALS) + [t for t in kept if t not in SPECIALS]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(token_to_id)

    @classmethod
    def load(cls, path: Path) -> "SmilesTokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls(d["token_to_id"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token_to_id": self.token_to_id}, f)

    # ---- encode/decode ----
    def encode(self, smi: str, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokenize(smi)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        toks: list[str] = []
        for i in ids:
            if i == self.bos_id:
                continue
            if i == self.eos_id or i == self.pad_id:
                break
            toks.append(self.id_to_token.get(i, ""))
        return "".join(toks)

    def n_unk(self, smiles: list[str]) -> int:
        """Diagnostic: how many tokens across `smiles` fall outside the vocab."""
        n = 0
        for s in smiles:
            for t in tokenize(s):
                if t not in self.token_to_id:
                    n += 1
        return n
