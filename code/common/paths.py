"""Shared paths for the Synthesis Challenge code (single-step retrosynthesis)."""
from __future__ import annotations
from pathlib import Path

# .../Synthesis-Challenge/code/common/paths.py  -> parents[2] = Synthesis-Challenge
CHALLENGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CHALLENGE_ROOT.parent

DATA_TRAIN = CHALLENGE_ROOT / "data_train.csv"
TEST_PRODUCTS = CHALLENGE_ROOT / "product_smiles_test.csv"
SAMPLE_SUBMISSION = CHALLENGE_ROOT / "sample_submission.csv"
EVAL_SCRIPT = CHALLENGE_ROOT / "top1_accuracy.py"

CODE_DIR = CHALLENGE_ROOT / "code"
ARTIFACTS_DIR = CHALLENGE_ROOT / "artifacts"
PREDICTIONS_DIR = CHALLENGE_ROOT / "predictions"
FINAL_DIR = PREDICTIONS_DIR / "final"

SEED = 1234
N_TEST = 10_000
N_HOLDOUT = 2_000

# Shared v0 artifacts every version depends on
HOLDOUT_IDX = ARTIFACTS_DIR / "v0" / "holdout_idx.json"
HOLDOUT_PRODUCTS = ARTIFACTS_DIR / "v0" / "holdout_products.csv"
HOLDOUT_TARGET = ARTIFACTS_DIR / "v0" / "holdout_target.csv"
LOOKUP_PKL = ARTIFACTS_DIR / "v0" / "lookup.pkl"
TOKENIZER_JSON = ARTIFACTS_DIR / "v0" / "tokenizer.json"
EDA_SUMMARY = ARTIFACTS_DIR / "v0" / "eda_summary.json"


def version_dirs(version: str) -> tuple[Path, Path, Path]:
    """Return (code, artifacts, predictions) dirs for the given version, creating
    the artifacts/predictions dirs if missing."""
    code = CODE_DIR / version
    art = ARTIFACTS_DIR / version
    preds = PREDICTIONS_DIR / version
    art.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    return code, art, preds
