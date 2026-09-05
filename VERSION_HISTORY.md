# Version History

A plain summary of what I tried, version by version. The score is top-1 accuracy: the
percent of test products where my single guess for the reactants was exactly right.

**Local** score is on a held-out slice of the training data. **Leaderboard** is the real
score from the server.

## v0 - Baselines

Three simple approaches to set a floor:
- Guess the product is its own reactant: 3.4%
- Nearest neighbor by molecular similarity: 9.8%
- Lookup table (reuse reactants from an identical product seen in training): 11.4%

The lookup table did better than expected because some test products also show up in
training, but it only ever covers part of the test set.

## v1 - Seq2seq model

A Transformer that reads the product and writes out the reactants, token by token.
Local score 27.7%. This became the main model for the rest of the project.

## v2 - Data augmentation

Trained on many different, equally valid ways of writing the same molecule, instead of
just one fixed way. This was the single biggest improvement: local score up to 29.5%.

## v3 - Forward check

Added a second model that checks: if you combine these predicted reactants, do you
actually get the product back? Used this to filter out bad guesses. Small gain, 29.9%
local.

## v4 - Ensemble

Combined predictions from two versions of the model instead of one. First real
submission: leaderboard score **30.1%**.

## v5 - Reranking

Instead of only using the model's top guess, used the forward-check model to rescore all
candidate answers and pick the most likely one. Local score 32.8%, though this number
was tuned on the same data it was tested on, so it is optimistic.

## v6 - Bigger ensemble

Trained a bigger model and combined 5 models in total, with a hand-tuned rule for
combining their guesses. Leaderboard score **35.1%**.

## v7 - Learned reranker (final)

Replaced the hand-tuned rule with a small model trained to combine and rank the
candidates, checked with proper cross-validation so the score can be trusted. Leaderboard
score **35.2%**, the final result.

## Where the remaining errors came from

After the challenge I checked the mistakes. About 18% of test products had no correct
answer anywhere among the generated candidates, so no amount of reranking could fix
those. Of the rest, roughly half were cases where the right answer was there but ranked
2nd or 3rd instead of 1st. So there was some real room to improve ranking, but the easy
gains had already been used up by v7.
