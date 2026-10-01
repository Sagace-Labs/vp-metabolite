# Model recipe

## Candidate generation

Apply 467 bundled transformation rules, radius-one reaction templates and learned
graph edits once to each parent. Templates are mined from a shuffled sample of
1,200 training pairs (seed 95) and retained only when they reproduce the example
product. Edit types require at least five training-pair occurrences.

Sanitise products, discard unchanged parents and products with fewer than three
heavy atoms, and merge routes by the InChIKey connectivity block. This identity
does not distinguish stereoisomers. Generation does not recursively expand
products; recorded relationships requiring several steps may be unreachable.

## Ranking

Each candidate has 26 route and structure features plus a 256-bin difference
count fingerprint (Morgan radius two). Structure features include size, mass,
lipophilicity, polar surface area, rings, donors, acceptors and parent similarity.
Route features retain source counts, phases and rule priors.

A 400-tree random forest uses minimum leaf size two, balanced bootstrap class
weights and seed zero. Recorded products are labelled one; other generated
candidates are labelled zero as a fitting proxy for unknown relationships.
Incomplete observations therefore affect both fitting and measured recall.
Forest scores order candidates and are not calibrated formation probabilities.
Ties follow connectivity-key order.

## Evaluation and deployment

`data/confirmation_split.json` fixes 1,474 training and 407 test parents with
cross-side Morgan similarity below 0.4. Evaluation rebuilds templates, edits and
the forest using training pairs only. Recall is averaged per parent against all
recorded products, including unreachable ones. Reachable recall reports the
candidate-generation ceiling; the baseline ranks the same candidates by their
maximum rule prior. The single seed does not measure sampling uncertainty.

Deployment rebuilds the generator and forest using all 3,117 parents and 7,081
pairs. The released weights contain the fitted forest, 302 verified templates
and 50 edit types. The generated version card records held-out measurements,
hashes and fitted environment; `requirements-reproduce.txt` pins dependencies.
