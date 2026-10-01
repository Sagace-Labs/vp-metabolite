# metabolite v1

_Generated from `manifest.toml` and `metrics.json`. Do not edit._

Released 2026-10-01 · signature 1

**Why this version.** First release of ranked candidate xenobiotic metabolite structures from five attributed directed-pair sources

## Outputs

| column | dtype | range | meaning |
|---|---|---|---|
| `metabolite_score` | float32 | 0.0–1.0 | Forest ranking score for a proposed metabolite; not a formation probability |

Missing values: An invalid input or a molecule with no generated candidates has no rows

## Performance

Protocol `metabolite-v1-fixed-similarity@1` — Frozen five-source directed-pair truth and parent-key split: 1,474 training parents and 407 held-out parents, with cross-side Morgan similarity below 0.4. Learned generators are rebuilt from training pairs only. One forest seed; metrics are parent-macro means.

Evaluated 2026-10-01 on n_train=1474, n_val=0, n_test=407.

| metric | mean | std | per seed |
|---|---|---|---|
| recall_at_1 | 0.208 | 0.000 | 0.208 |
| recall_at_5 | 0.540 | 0.000 | 0.540 |
| recall_at_10 | 0.628 | 0.000 | 0.628 |
| recall_at_20 | 0.715 | 0.000 | 0.715 |
| reachable_recall | 0.786 | 0.000 | 0.786 |
| candidates_per_parent | 283.936 | 0.000 | 283.936 |
| prior_recall_at_10 | 0.448 | 0.000 | 0.448 |

> Comparable only with metrics carrying the same protocol id.

## Data

Five-source xenobiotic parent-metabolite relationships — MetXBioDB, ChEMBL metabolism, e-Drug3D, observed MetSim and NCATS Inxight/GSRS. Retrieved 2026-09-21, licensed CC-BY-SA-4.0, redistributed here.

`7081` directed pairs, table SHA-256 `af54013d42b79022…`

Verify the bundled release with `python -m vp_metabolite.data verify`.

## Model

verified-rule and graph-edit generator with random-forest ranker on `route counts, product descriptors, parent-product deltas and difference fingerprint` features. Shipped weights: all 3117 parents; 1,200 train-pair radius-one template sample and train-only edit vocabulary

`weights.joblib` SHA-256 `40b5b1ee3e72a65c…`

## Provenance

Environment: python 3.11.11, rdkit 2026.03.5, sklearn 1.9.0.

The bundled source and table hashes pin this directed-pair release; refits also depend on the recorded environment.
