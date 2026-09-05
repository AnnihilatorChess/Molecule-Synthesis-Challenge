"""Submission guardrails.

Guarantee every emitted line is non-empty and canonicalizes to >= 1 fragment, that
the file has exactly the expected number of lines, has no header, and ends with a
single trailing newline (matching sample_submission.csv).  Any row that would be
empty/invalid is replaced by the identity fallback (the product itself, which is
always a valid SMILES), so the grader never drops a row or crashes.
"""
from __future__ import annotations

from pathlib import Path

from .canon import canon_frag_list


def _fallback(product: str) -> str:
    """Identity fallback: the (canonicalized if possible) product as its own reactant."""
    frags = canon_frag_list(product)
    if frags:
        return ".".join(sorted(set(frags)))
    return product.strip() or "C"  # last-ditch: a trivially valid SMILES


def safe_lines(preds: list[str | None], products: list[str]) -> list[str]:
    """Clean predictions, filling invalid/empty rows with the identity fallback."""
    assert len(preds) == len(products), "preds/products length mismatch"
    out: list[str] = []
    for pred, prod in zip(preds, products):
        if pred and isinstance(pred, str) and canon_frag_list(pred):
            out.append(pred.strip())
        else:
            out.append(_fallback(prod))
    return out


def write_submission(lines: list[str], path: Path, expected_n: int | None = None) -> None:
    if expected_n is not None:
        assert len(lines) == expected_n, f"expected {expected_n} lines, got {len(lines)}"
    assert all(ln and ln.strip() for ln in lines), "empty line in submission"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(ln.strip() for ln in lines) + "\n")


def validate_file(path: Path, expected_n: int) -> None:
    """Re-read a written submission and assert shape + validity."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    assert len(lines) == expected_n, f"{path.name}: {len(lines)} lines, expected {expected_n}"
    bad = [i for i, ln in enumerate(lines) if not canon_frag_list(ln)]
    assert not bad, f"{path.name}: {len(bad)} lines canonicalize to nothing (first idx {bad[:3]})"
