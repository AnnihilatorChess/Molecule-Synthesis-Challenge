# Plan

Notes I wrote before starting, to think through the task and decide on an approach.

## Task

Single-step retrosynthesis: given the product of a chemical reaction, predict the reactants
that made it. Training data is 40,000 real reactions (reactants and product both given).
The test set only gives 10,000 products, and I had to predict the reactants for each.

Scoring: one guess per product, correct only if it exactly matches the true reactants.

## What I found looking at the data

- All training and test molecules are valid and parse correctly.
- Molecules are fairly small and short, so a generous length limit is safe.
- Most reactions have just one reactant molecule, but a good third have two or more.
- About 22% of the test products also show up in the training data. But the same product can
  map to different reactants across different rows, so even an exact match in the lookup table
  is only right part of the time. A realistic score is somewhere between 15% and 35%, well
  below what similar academic datasets report, since this data is noisier.

## Approach

1. **Simple baselines first**: guessing the product is its own reactant, nearest-neighbor
   lookup by molecule similarity, and an exact-match lookup table. These give a safety net and
   a sense of the floor.
2. **Main model**: a sequence-to-sequence Transformer that reads the product and writes out the
   reactants, token by token. Training on many different valid text versions of the same
   molecule (data augmentation) was expected to be the biggest lever. Beam search generates
   several candidate answers instead of just one.
3. **Forward check**: a second model that reads candidate reactants and predicts the product,
   used to check whether a candidate actually explains the real product.
4. **Combine approaches**: fall back through lookup, then the neural model, then nearest
   neighbor, so every row always gets a valid guess.
5. Pick whichever version scores best on a held-out slice of the training data, to avoid
   guessing blind on the leaderboard.

## Local validation

I held out 2,000 training rows and scored every model version against them with the official
scoring script, so local numbers would be trustworthy and I would not need to burn real
submissions just to compare ideas.

## Running the code

```bash
python code/run_all.py   # baselines, seq2seq model, ensemble
```

See `VERSION_HISTORY.md` for the reranking stage (v5 to v7) and the full set of results.
