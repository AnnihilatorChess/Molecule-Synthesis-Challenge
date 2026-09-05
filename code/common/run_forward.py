"""Train the FORWARD model (reactants -> product) used for round-trip reranking.

Same architecture/schedule as the retro model, just with source/target swapped and the
reactants augmented.  Checkpoint is saved to artifacts/<version>/fwd_ckpt_best.pt.
"""
from __future__ import annotations

from pathlib import Path

from . import data, paths
from .data import read_lines
from .tokenizer import SmilesTokenizer
from .trainer import train_seq2seq


def run(version: str = "v3", augment: bool = True, epochs: int = 30,
        batch_size: int = 128, accum: int = 1, eval_every: int = 3,
        num_workers: int = 0, model_overrides: dict | None = None) -> dict:
    _, art, _ = paths.version_dirs(version)
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)

    # forward direction: source = reactants (LEFT), target = product (RIGHT)
    train_pairs = [(r, p) for p, r in data.train_pairs_canon()]
    val_src = read_lines(paths.HOLDOUT_TARGET)          # reactants (raw)
    val_tgt = read_lines(paths.HOLDOUT_PRODUCTS)        # products (raw)
    assert len(val_src) == len(val_tgt)

    best = train_seq2seq(
        train_pairs, val_src, val_tgt, tok, art, augment=augment, epochs=epochs,
        batch_size=batch_size, accum=accum, eval_every=eval_every,
        num_workers=num_workers, model_overrides=model_overrides,
        tag=f"forward_{version}", log_name="forward_run.log")

    # rename the checkpoint so it doesn't collide with the retro model's ckpt_best.pt
    src_ck = art / "ckpt_best.pt"
    dst_ck = art / "fwd_ckpt_best.pt"
    if src_ck.exists():
        src_ck.replace(dst_ck)
    print(f"[forward_{version}] best val_top1={best['top1']:.4f} -> {dst_ck.name}")
    return best
