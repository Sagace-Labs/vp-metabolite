# vp-metabolite

Ranks candidate xenobiotic metabolite structures from a parent SMILES string.
The scores order generated structures; they are not formation probabilities.

## Install

Install Git LFS for the bundled weights, then run:

    git lfs install
    pip install git+https://github.com/Sagace-Labs/vp-core.git@e49310b74a164bc30974053af2ab1a5236f46161
    pip install git+https://github.com/Sagace-Labs/vp-metabolite.git@v1

## Use

    from vp_metabolite import predict
    candidates = predict(["CC(=O)Nc1ccc(O)cc1"], limit=10)

Returns one row per candidate, indexed by input position, rank, product SMILES
and InChIKey connectivity block. `metabolite_score` is the declared output
column. An invalid input or one with no candidates returns no rows. Pin the
artifact with `version="v1"`; inspect available artifacts with `versions()`.

## Current version

**v1**, signature 1, measured under `metabolite-v1-fixed-similarity@1`.
The generated [card](src/vp_metabolite/versions/v1/CARD.md) records the held-out
metrics, dataset hash and environment.
See [MODEL.md](MODEL.md) for the generator, fitting targets and evaluation recipe.

## Data

The bundled release has 7,081 directed parent-product relationships from five
attributed sources, with per-assertion provenance and source snapshots under
`data/release/`. Unrecorded candidates are unknown, not established negatives.
Some recorded links need more than one transformation; `reachable_recall` in the
card measures the generator's ceiling against the bundled relationships.
The candidate rules are bundled under `src/vp_metabolite/resources/`.
See [data/README.md](data/README.md) and [LICENSE-DATA](LICENSE-DATA).

## Retrain

Use Python 3.11.11 and the pinned `requirements-reproduce.txt`; run from a
repository clone containing `data/release/`.

    pip install -r requirements-reproduce.txt
    pip install -e . --no-deps
    python -m vp_metabolite.data verify
    python -m vp_metabolite.train --version v2 --reason "why this version exists"
    python -m vp_metabolite.evaluate --version v2

Training rebuilds learned templates and edits from the pair table. Evaluation
rebuilds them from the frozen training parents only and tests on held-out
parents separated by molecular similarity. It records parent-macro recall,
reachable recall, candidate count and a rule-prior baseline.
The one-seed standard deviation in the card is not an uncertainty interval.
Pass `--cache-dir .vp-metabolite-cache` to either command to resume candidate
generation after an interruption.

## Licence

Code, fitted weights and bundled transformation rules are GPL-3.0 ([LICENSE](LICENSE)).
The processed pair tables are CC BY-SA 4.0 ([LICENSE-DATA](LICENSE-DATA)).

## Cite

See [CITATION.cff](CITATION.cff) and the source attributions in
[LICENSE-DATA](LICENSE-DATA).
