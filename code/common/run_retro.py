"""Train a retrosynthesis (product->reactants) seq2seq model, then decode + score.

Shared by v1 (no augmentation) and v2 (augmentation + TTA + wider beam).  Caches the
per-row beam candidate lists (holdout + test) to artifacts/<version>/candidates.pkl so
v3/v4 can reuse them without re-decoding.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from . import data, paths
from .data import read_lines
from .decode import beam_candidates, load_model
from .finalize import safe_lines, write_submission
from .local_metric import score_files
from .tokenizer import SmilesTokenizer
from .trainer import train_seq2seq


def _top1_preds(candidates):
    """Best (highest-score) candidate string per row; None if no valid candidate."""
    return [c[0][0] if c else None for c in candidates]


def run(version: str, augment: bool, epochs: int, beam_width: int, tta_k: int,
        batch_size: int = 128, accum: int = 1, eval_every: int = 2,
        num_workers: int = 0, model_overrides: dict | None = None,
        seed: int = 1234) -> dict:
    t0 = time.time()
    _, art, preds = paths.version_dirs(version)
    tok = SmilesTokenizer.load(paths.TOKENIZER_JSON)

    train_pairs = data.train_pairs_canon()                 # (product, reactants)
    val_src = data.holdout_products_canon()                # canonical products
    val_tgt = read_lines(paths.HOLDOUT_TARGET)             # raw reactant strings (LEFT)
    assert len(val_src) == len(val_tgt)

    best = train_seq2seq(
        train_pairs, val_src, val_tgt, tok, art, augment=augment, epochs=epochs,
        batch_size=batch_size, accum=accum, eval_every=eval_every,
        num_workers=num_workers, model_overrides=model_overrides, seed=seed,
        tag=f"retro_{version}")

    model, _ = load_model(art / "ckpt_best.pt")

    # holdout candidates + score
    cand_h = beam_candidates(model, tok, val_src, beam_width=beam_width, tta_k=tta_k)
    preds_h = safe_lines(_top1_preds(cand_h), val_src)
    hpath = preds / f"{version}_holdout_pred.csv"
    write_submission(preds_h, hpath, expected_n=len(val_src))
    holdout_top1 = score_files(hpath)

    # test candidates
    test_prods = data.load_test_canon()["canon"]
    cand_t = beam_candidates(model, tok, test_prods, beam_width=beam_width, tta_k=tta_k)
    preds_t = safe_lines(_top1_preds(cand_t), test_prods)
    write_submission(preds_t, preds / f"{version}_test_pred.csv", expected_n=len(test_prods))

    with open(art / "candidates.pkl", "wb") as f:
        pickle.dump({"holdout": cand_h, "test": cand_t}, f)
    summary = {"version": version, "augment": augment, "beam_width": beam_width,
               "tta_k": tta_k, "train_best_val_top1": best["top1"],
               "holdout_top1": holdout_top1, "minutes": round((time.time() - t0) / 60, 1)}
    with open(art / "result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{version}] holdout top-1 = {holdout_top1:.4f}  (train-greedy best {best['top1']:.4f})")
    return summary
