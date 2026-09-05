"""Final integrity gate for predictions/final/submission.csv.

Asserts: exactly N_TEST non-empty lines; every line canonicalizes to >=1 fragment; and the
OFFICIAL top1_accuracy.calculate_top_1_accuracy can ingest the file end-to-end (run against
itself, which must return 1.0) — proving the grader will parse it without error.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import paths  # noqa: E402
from common.finalize import validate_file  # noqa: E402


def _load_official():
    spec = importlib.util.spec_from_file_location("top1_accuracy", paths.EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    final = paths.FINAL_DIR / "submission.csv"
    if not final.exists():
        print(f"[verify] FAIL: {final} does not exist")
        return 1
    validate_file(final, paths.N_TEST)
    official = _load_official()
    self_acc = official.calculate_top_1_accuracy(str(final), str(final))
    assert abs(self_acc - 1.0) < 1e-9, f"official self-score {self_acc} != 1.0 (parse issue)"
    print(f"[verify] OK: {final}")
    print(f"[verify] {paths.N_TEST} valid lines; official script ingests file cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
