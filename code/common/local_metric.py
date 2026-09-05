"""Local top-1 scoring.

Two routes, asserted to agree on v0:
  - score_files(): shells out to the OFFICIAL `top1_accuracy.py` (byte-faithful grader).
  - score_lists(): fast in-process reimplementation for the model-selection inner loop.

Both implement the same rule: a row is correct iff the canonical reactant SET of the
prediction equals that of the truth.  Rows where the prediction is empty/NaN are dropped
(matching the official `dropna()`), so always emit a non-empty best guess in real runs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import paths
from .canon import canon_set


def score_lists(preds: list[str], trues: list[str]) -> tuple[float, int, int]:
    """Return (top1_accuracy, n_correct, n_scored). Drops rows with empty prediction."""
    assert len(preds) == len(trues), "preds/trues length mismatch"
    correct = 0
    scored = 0
    for p, t in zip(preds, trues):
        if p is None or (isinstance(p, str) and not p.strip()):
            continue  # mimic dropna()
        scored += 1
        if canon_set(p) == canon_set(t):
            correct += 1
    acc = correct / scored if scored else 0.0
    return acc, correct, scored


def score_files(submission_path: Path, target_path: Path | None = None) -> float:
    """Invoke the official evaluation script and parse the printed accuracy."""
    target_path = target_path or paths.HOLDOUT_TARGET
    out = subprocess.run(
        [sys.executable, str(paths.EVAL_SCRIPT),
         "--submission", str(submission_path),
         "--target", str(target_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])
