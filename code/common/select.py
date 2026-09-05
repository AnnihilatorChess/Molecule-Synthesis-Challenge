"""Auto-select the best submission by held-out top-1 (guarantees no regression).

Scores every predictions/<v>/*_holdout_pred.csv with the OFFICIAL metric, picks the max,
and copies the corresponding *_test_pred.csv to predictions/final/submission.csv.  Even
if every neural model underperforms, the retrieval baselines remain in the pool, so the
shipped file can only be >= the best baseline on the held-out slice.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import paths
from .finalize import validate_file
from .local_metric import score_files


def select_best(expected_n: int = None) -> dict:
    expected_n = expected_n or paths.N_TEST
    holdout_preds = sorted(paths.PREDICTIONS_DIR.glob("v*/*_holdout_pred.csv"))
    board = []
    for hp in holdout_preds:
        tp = hp.with_name(hp.name.replace("_holdout_pred.csv", "_test_pred.csv"))
        if not tp.exists():
            continue
        try:
            score = score_files(hp)
        except Exception as e:  # never let one bad file sink selection
            print(f"[select] skip {hp.name}: {e}")
            continue
        board.append({"name": f"{hp.parent.name}/{hp.stem}", "holdout_top1": score,
                      "test_pred": str(tp)})
    board.sort(key=lambda d: d["holdout_top1"], reverse=True)
    assert board, "no scorable holdout/test prediction pairs found"

    best = board[0]
    paths.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final = paths.FINAL_DIR / "submission.csv"
    shutil.copyfile(best["test_pred"], final)
    validate_file(final, expected_n)

    result = {"selected": best["name"], "holdout_top1": best["holdout_top1"],
              "final": str(final), "leaderboard": board}
    with open(paths.PREDICTIONS_DIR / "leaderboard.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[select] leaderboard:")
    for d in board:
        print(f"    {d['holdout_top1']:.4f}  {d['name']}")
    print(f"[select] -> {final}  (={best['name']}, holdout {best['holdout_top1']:.4f})")
    return result
