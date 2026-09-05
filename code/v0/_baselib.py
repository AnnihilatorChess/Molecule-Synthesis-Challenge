"""Shared helper for v0 baselines: clean + write holdout/test predictions and score."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import paths, local_metric  # noqa: E402
from common.finalize import safe_lines, write_submission  # noqa: E402


def write_and_score(
    name: str,
    holdout_preds: list[str | None],
    holdout_prods: list[str],
    test_preds: list[str | None],
    test_prods: list[str],
) -> float:
    preds_dir = paths.PREDICTIONS_DIR / "v0"
    h = safe_lines(holdout_preds, holdout_prods)
    write_submission(h, preds_dir / f"{name}_holdout_pred.csv", expected_n=len(holdout_prods))
    score = local_metric.score_files(preds_dir / f"{name}_holdout_pred.csv")

    t = safe_lines(test_preds, test_prods)
    write_submission(t, preds_dir / f"{name}_test_pred.csv", expected_n=len(test_prods))

    with open(paths.ARTIFACTS_DIR / "v0" / f"{name}_score.json", "w") as f:
        json.dump({"holdout_top1": score}, f)
    print(f"[{name}] holdout top-1 = {score:.4f}")
    return score
