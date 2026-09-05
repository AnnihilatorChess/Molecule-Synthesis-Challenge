"""Dataset + collate for seq2seq training.

A `PairDataset` holds (source, target) SMILES pairs and a shared tokenizer.  With
`augment_src=True` the source is re-randomized on every __getitem__ (on-the-fly
augmentation); the target is kept as-is (canonical, fragment-sorted) so the decoder
learns one consistent generation order.
"""
from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import Dataset

from .augment import randomize_smiles


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tok, max_len: int = 64,
                 augment_src: bool = False):
        self.pairs = pairs
        self.tok = tok
        self.max_len = max_len
        self.augment_src = augment_src

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        src, tgt = self.pairs[i]
        if self.augment_src:
            src = randomize_smiles(src)
        src_ids = self.tok.encode(src)[: self.max_len]
        tgt_ids = self.tok.encode(tgt)[: self.max_len]
        return np.asarray(src_ids, dtype=np.int64), np.asarray(tgt_ids, dtype=np.int64)


def make_collate(pad_id: int):
    def collate(batch):
        srcs, tgts = zip(*batch)
        Ts = max(len(s) for s in srcs)
        Tt = max(len(t) for t in tgts)
        B = len(batch)
        src = np.full((B, Ts), pad_id, dtype=np.int64)
        tgt = np.full((B, Tt), pad_id, dtype=np.int64)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, : len(s)] = s
            tgt[i, : len(t)] = t
        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)
        return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()
    return collate


def encode_src_batch(smiles: list[str], tok, max_len: int, device) -> torch.Tensor:
    """Encode a list of source SMILES into a padded (B, T) tensor for inference."""
    ids = [tok.encode(s)[:max_len] for s in smiles]
    T = max(len(x) for x in ids)
    out = np.full((len(ids), T), tok.pad_id, dtype=np.int64)
    for i, x in enumerate(ids):
        out[i, : len(x)] = x
    return torch.from_numpy(out).to(device)
