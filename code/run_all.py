"""One-shot orchestrator: v0 -> v1 -> v2 -> v3 -> v4 -> final selection -> verify.

Each stage runs as an isolated subprocess; a failure is logged and the pipeline continues
(graceful degradation to the retrieval cascade).  Stages whose sentinel output already
exists are skipped unless --force is given, so the run is resumable.

Usage:
    python code/run_all.py            # resume / run missing stages
    python code/run_all.py --force    # rerun everything from scratch
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import paths  # noqa: E402

A = paths.ARTIFACTS_DIR
P = paths.PREDICTIONS_DIR

# (label, script relative to code/, sentinel file that means "already done")
STAGES = [
    ("v0:make_holdout", "v0/make_holdout.py", paths.HOLDOUT_IDX),
    ("v0:orientation", "v0/check_orientation.py", None),  # always (cheap assertion)
    ("v0:eda", "v0/eda.py", paths.EDA_SUMMARY),
    ("v0:lookup", "v0/baseline_lookup.py", A / "v0" / "lookup_score.json"),
    ("v0:identity", "v0/baseline_identity.py", A / "v0" / "identity_score.json"),
    ("v0:knn", "v0/baseline_knn.py", A / "v0" / "knn_score.json"),
    ("v1", "v1/run.py", A / "v1" / "result.json"),
    ("v2", "v2/run.py", A / "v2" / "result.json"),
    ("v3", "v3/run.py", A / "v3" / "result.json"),
    ("v4", "v4/run.py", A / "v4" / "result.json"),
]


def run_stage(label, script, sentinel, force) -> bool:
    if sentinel is not None and sentinel.exists() and not force:
        print(f"[run_all] skip {label} (sentinel {sentinel.name} exists)")
        return True
    print(f"[run_all] === {label} === ({script})")
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(HERE / script)]).returncode
    dt = time.time() - t0
    if rc != 0:
        print(f"[run_all] !! {label} FAILED rc={rc} after {dt:.0f}s (continuing)")
        return False
    print(f"[run_all] ok {label} in {dt:.0f}s")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for label, script, sentinel in STAGES:
        try:
            run_stage(label, script, sentinel, args.force)
        except Exception as e:
            print(f"[run_all] !! {label} raised {e!r} (continuing)")

    # ensure a final submission exists even if v4 failed
    final = paths.FINAL_DIR / "submission.csv"
    if not final.exists():
        print("[run_all] v4 did not produce final submission; running selector directly")
        try:
            subprocess.run([sys.executable, str(HERE / "select_final.py")])
        except Exception as e:
            print(f"[run_all] !! selector failed: {e!r}")

    # verify
    subprocess.run([sys.executable, str(HERE / "verify_submission.py")])
    print(f"[run_all] ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
