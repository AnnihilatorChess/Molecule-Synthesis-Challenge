# Molecule Synthesis Challenge

JKU Linz "AI for Life Sciences" course challenge. Placed **4th out of 134** participants.

## The task

Single-step retrosynthesis: given the product of a chemical reaction (as a SMILES string),
predict the reactant(s) that were combined to make it. This is the reverse of normal chemistry:
instead of "what do these ingredients make", it is "what ingredients make this".

Scoring was **top-1 exact match**: for each of the 10,000 test products, one prediction was
allowed, and it had to exactly match the true set of reactants.

## My approach

**1. Baselines first.** A simple lookup table (if this exact product was seen in training, reuse
its reactants) already got about 11% accuracy, because some test products repeat in training.
This became the safety net for the final system.

**2. Neural model.** A sequence-to-sequence Transformer that reads the product SMILES and writes
out the reactant SMILES, token by token. Two things gave the biggest gains:

- **Data augmentation**: feeding the model different, equally valid ways of writing the same
  molecule during training. This was the single biggest improvement, more than any architecture
  change.
- **Beam search**: instead of generating one best guess, the model considers several candidate
  answers and picks the most likely one.

**3. Combining several models.** I trained multiple versions of the model (different sizes,
different random seeds) and combined their candidate answers, plus a second "forward" model that
checks whether a candidate's reactants would actually recreate the product. A learned reranker
(a small logistic regression model) then picked the final answer from all candidates, using
things like how many models agreed, whether the forward model approved, and whether the atoms
balance out.

**4. Error analysis.** After the challenge, I broke down where the errors came from: about 18%
of test products had no correct answer anywhere in the candidate pool at all (nothing to be done
there), around 45% of the time the correct answer was in the pool but ranked below another
candidate, and among those near-misses, roughly half had the correct answer sitting at rank 2 or
3. So a meaningful chunk of the remaining errors were "close, but the ranking was wrong" rather
than the model being clueless. Full details are in [VERSION_HISTORY.md](VERSION_HISTORY.md).

## Result

| Stage | Approach | Score |
|---|---|---|
| Baseline | exact-match lookup | 11.4% (local) |
| v4 | single seq2seq model + ensemble | 30.1% (leaderboard) |
| v6 | 5-model ensemble + reranking | 35.1% (leaderboard) |
| v7 | + learned reranker | **35.2%** (leaderboard, best) |

## Files

- `code/` - all versions, from baselines (`v0`) to the final learned reranker (`v7`)
- `code/analyze_errors.py` - the post-challenge error analysis
- `VERSION_HISTORY.md` - detailed log of every version, its results, and what I learned from it
- `CHALLENGE_PLAN.md` - the plan and data analysis written before starting
- `report.pdf` - final write-up submitted for grading
- `presentation.tex` - slides for the challenge presentation
- `predictions/final/submission.csv` - the submitted predictions
- `top1_accuracy.py` - the official evaluation script

## What is not included

The training data (40,000 reactions, provided by the course) and intermediate prediction files
from earlier versions are left out to keep the repo small. Only the final submission is kept.

## Dependencies

PyTorch, RDKit, numpy, pandas, scikit-learn.
