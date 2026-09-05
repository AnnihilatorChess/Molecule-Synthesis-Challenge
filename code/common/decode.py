"""Load a trained seq2seq checkpoint and produce ranked, valid candidate sets.

`beam_candidates` returns, per source SMILES, a list of (canonical_target_string, score)
deduplicated at the canonical-set level, optionally with test-time augmentation (TTA):
each source is decoded from K randomized SMILES of the same molecule and the candidate
sets are pooled and vote-weighted (score = log sum exp of beam scores across variants).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch

from .augment import randomize_smiles
from .canon import canon_join
from .seq2seq import Seq2Seq
from .traindata import encode_src_batch


def load_model(ckpt_path: Path, device: str = "cuda") -> tuple[Seq2Seq, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Seq2Seq(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def _decode_batch(model, tok, srcs, beam_width, max_len, length_penalty, device):
    s = encode_src_batch(srcs, tok, max_len, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hyps = model.beam_search(s, tok.bos_id, tok.eos_id, beam_width=beam_width,
                                 max_len=max_len, length_penalty=length_penalty)
    # hyps[i] = list of (token_ids, score)
    out = []
    for hyp_list in hyps:
        cand = []
        for ids, score in hyp_list:
            cs = canon_join(tok.decode(ids))
            if cs:
                cand.append((cs, score))
        out.append(cand)
    return out


@torch.no_grad()
def forward_loglik(model, tok, pairs: list[tuple[str, str]], max_len: int = 64,
                   batch_size: int = 256, length_norm: bool = True,
                   device: str = "cuda") -> list[float]:
    """Teacher-forced avg log P(target | source) for each (source, target) pair.
    Used to rerank candidate reactant sets by how well the FORWARD model regenerates
    the product (round-trip likelihood)."""
    import numpy as np
    out: list[float] = []
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b : b + batch_size]
        srcs = [tok.encode(s)[:max_len] for s, _ in chunk]
        tgts = [tok.encode(t)[:max_len] for _, t in chunk]
        Ts = max(len(s) for s in srcs)
        Tt = max(len(t) for t in tgts)
        src = np.full((len(chunk), Ts), tok.pad_id, np.int64)
        tgt = np.full((len(chunk), Tt), tok.pad_id, np.int64)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, : len(s)] = s
            tgt[i, : len(t)] = t
        src = torch.from_numpy(src).to(device)
        tgt = torch.from_numpy(tgt).to(device)
        tin, tout = tgt[:, :-1], tgt[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(src, tin)
        import torch.nn.functional as F
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, tout.unsqueeze(-1)).squeeze(-1)  # (B, T)
        mask = (tout != tok.pad_id).float()
        summ = (tok_lp * mask).sum(1)
        if length_norm:
            summ = summ / mask.sum(1).clamp(min=1)
        out.extend(summ.tolist())
    return out


@torch.no_grad()
def forward_predict(model, tok, sources: list[str], max_len: int = 64,
                    batch_size: int = 512, device: str = "cuda") -> dict[str, str]:
    """Greedy-decode each UNIQUE source string once -> canonical predicted target.
    Used for round-trip checking (reactants -> product).  Returns {source: pred_canon}."""
    uniq = sorted(set(sources))
    out: dict[str, str] = {}
    for b in range(0, len(uniq), batch_size):
        chunk = uniq[b : b + batch_size]
        s = encode_src_batch(chunk, tok, max_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.greedy_decode(s, tok.bos_id, tok.eos_id, max_len=max_len)
        for src, seq in zip(chunk, ids):
            out[src] = canon_join(tok.decode(seq)) or ""
    return out


def beam_candidates(model, tok, products: list[str], beam_width: int = 10,
                    max_len: int = 64, length_penalty: float = 0.6, tta_k: int = 1,
                    batch_size: int = 256, device: str = "cuda") -> list[list[tuple[str, float]]]:
    """Return per-product ranked candidate (reactants_str, score) lists (valid, set-deduped)."""
    # Build the flat list of (product_index, source_string) to decode.
    flat_idx, flat_src = [], []
    for i, p in enumerate(products):
        variants = [p]
        for _ in range(max(0, tta_k - 1)):
            variants.append(randomize_smiles(p))
        for v in variants:
            flat_idx.append(i)
            flat_src.append(v)

    # Decode in batches.
    per_source: list[list[tuple[str, float]]] = []
    for b in range(0, len(flat_src), batch_size):
        per_source.extend(_decode_batch(
            model, tok, flat_src[b : b + batch_size], beam_width, max_len,
            length_penalty, device))

    # Pool candidates back per product (vote-weighted by exp(score)).
    pooled: list[dict[str, float]] = [defaultdict(float) for _ in products]
    for src_i, cand in zip(flat_idx, per_source):
        for cs, score in cand:
            pooled[src_i][cs] += math.exp(score)

    results = []
    for d in pooled:
        ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        results.append([(cs, math.log(w)) for cs, w in ranked])
    return results
