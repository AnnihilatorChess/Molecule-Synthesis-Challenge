"""Encoder-decoder Transformer for SMILES->SMILES (retrosynthesis / forward synthesis).

Pre-norm blocks, learned positional embeddings, tied token embeddings + output head
(reusing the Generation Challenge v2 style).  The decoder adds a cross-attention sublayer
over the encoder memory.

Molecules here are tiny (<=25 tokens), so beam search uses full recompute per step rather
than a KV-cache — far simpler and still cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """Multi-head attention usable for encoder self / decoder self / cross attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.nh = n_heads
        self.hd = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x_q, x_kv, attn_mask=None, is_causal=False):
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]
        q = self.q(x_q).view(B, Tq, self.nh, self.hd).transpose(1, 2)
        k = self.k(x_kv).view(B, Tk, self.nh, self.hd).transpose(1, 2)
        v = self.v(x_kv).view(B, Tk, self.nh, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj(out)


class _MLP(nn.Sequential):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim), nn.Dropout(dropout),
        )


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)

    def forward(self, x, src_mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, attn_mask=src_mask, is_causal=False)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = MHA(dim, n_heads, dropout)
        self.ln_x = nn.LayerNorm(dim)
        self.cross_attn = MHA(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)

    def forward(self, x, memory, mem_mask):
        h = self.ln1(x)
        x = x + self.self_attn(h, h, is_causal=True)
        x = x + self.cross_attn(self.ln_x(x), memory, attn_mask=mem_mask, is_causal=False)
        x = x + self.mlp(self.ln2(x))
        return x


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 64,
        dim: int = 256,
        n_heads: int = 8,
        n_enc: int = 4,
        n_dec: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.enc_pos = nn.Embedding(max_len, dim)
        self.dec_pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.enc_blocks = nn.ModuleList(
            [EncoderBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(n_enc)]
        )
        self.dec_blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(n_dec)]
        )
        self.enc_ln = nn.LayerNorm(dim)
        self.dec_ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tie
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _src_additive_mask(self, src):
        """(B,1,1,Tsrc) additive mask: 0 where real token, -inf where pad."""
        pad = src == self.pad_id  # (B, Tsrc)
        mask = torch.zeros_like(pad, dtype=torch.float32)
        mask = mask.masked_fill(pad, float("-inf"))
        return mask[:, None, None, :]

    def encode(self, src):
        B, T = src.shape
        pos = torch.arange(T, device=src.device)
        h = self.drop(self.tok_emb(src) + self.enc_pos(pos))
        src_mask = self._src_additive_mask(src)
        for blk in self.enc_blocks:
            h = blk(h, src_mask)
        return self.enc_ln(h), src_mask

    def decode(self, tgt_in, memory, mem_mask):
        B, T = tgt_in.shape
        pos = torch.arange(T, device=tgt_in.device)
        h = self.drop(self.tok_emb(tgt_in) + self.dec_pos(pos))
        for blk in self.dec_blocks:
            h = blk(h, memory, mem_mask)
        return self.head(self.dec_ln(h))

    def forward(self, src, tgt_in):
        memory, mem_mask = self.encode(src)
        return self.decode(tgt_in, memory, mem_mask)

    # ---------------- inference ----------------
    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=64):
        self.eval()
        memory, mem_mask = self.encode(src)
        B = src.shape[0]
        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.decode(seqs, memory, mem_mask)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            nxt = nxt.masked_fill(done[:, None], eos_id)
            seqs = torch.cat([seqs, nxt], dim=1)
            done = done | (nxt.squeeze(1) == eos_id)
            if done.all():
                break
        return seqs[:, 1:].tolist()  # strip bos

    @torch.no_grad()
    def beam_search(self, src, bos_id, eos_id, beam_width=10, max_len=64,
                    length_penalty=0.6, n_best=None):
        """Batched beam search via full recompute. Returns, per source, a list of
        (token_ids, normalized_score) sorted best-first (length up to n_best)."""
        self.eval()
        n_best = n_best or beam_width
        device = src.device
        B = src.shape[0]
        W = beam_width
        memory, mem_mask = self.encode(src)                       # (B,Ts,D),(B,1,1,Ts)
        Ts = memory.shape[1]
        memory = memory[:, None].expand(B, W, Ts, self.dim).reshape(B * W, Ts, self.dim)
        mem_mask = mem_mask[:, None].expand(B, W, 1, 1, Ts).reshape(B * W, 1, 1, Ts)

        seqs = torch.full((B * W, 1), bos_id, dtype=torch.long, device=device)
        beam_lp = torch.full((B, W), float("-inf"), device=device)
        beam_lp[:, 0] = 0.0
        beam_lp = beam_lp.reshape(B * W)
        done = torch.zeros(B * W, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            logits = self.decode(seqs, memory, mem_mask)[:, -1, :]        # (B*W, V)
            logp = F.log_softmax(logits.float(), dim=-1)
            V = logp.shape[-1]
            # finished beams: freeze (only emit pad with 0 added logprob)
            frozen = torch.full_like(logp, float("-inf"))
            frozen[:, self.pad_id] = 0.0
            logp = torch.where(done[:, None], frozen, logp)
            cand = beam_lp[:, None] + logp                                # (B*W, V)
            cand = cand.view(B, W * V)
            top_scores, top_idx = cand.topk(W, dim=-1)                    # (B, W)
            parent = top_idx // V                                         # (B, W) in [0,W)
            token = top_idx % V                                           # (B, W)
            flat_parent = (parent + torch.arange(B, device=device)[:, None] * W).reshape(-1)
            seqs = torch.cat([seqs[flat_parent], token.reshape(-1, 1)], dim=1)
            beam_lp = top_scores.reshape(-1)
            done = done[flat_parent] | (token.reshape(-1) == eos_id)
            if done.all():
                break

        seqs = seqs.view(B, W, -1)
        beam_lp = beam_lp.view(B, W)
        results = []
        for b in range(B):
            hyps = []
            for w in range(W):
                ids = seqs[b, w, 1:].tolist()  # strip bos
                if eos_id in ids:
                    ids = ids[: ids.index(eos_id)]
                ln = max(len(ids), 1)
                penalty = ((5 + ln) / 6) ** length_penalty
                hyps.append((ids, float(beam_lp[b, w]) / penalty))
            hyps.sort(key=lambda x: x[1], reverse=True)
            results.append(hyps[:n_best])
        return results


def build_model(tok, **overrides):
    cfg = dict(vocab_size=tok.vocab_size, max_len=64, dim=256, n_heads=8,
               n_enc=4, n_dec=4, mlp_ratio=4, dropout=0.1, pad_id=tok.pad_id)
    cfg.update(overrides)
    return Seq2Seq(**cfg), cfg
