# Data

`release/` contains the frozen directed-pair table, its source observations,
the source snapshots and a manifest with SHA-256 hashes. A pair is keyed by
the parent and product InChIKey connectivity blocks. The table records known
relationships; absent pairs are not known negatives.

`confirmation_split.json` fixes the parent identities used for held-out
evaluation. It records excluded identities, a train/test Morgan similarity
threshold of 0.4, and a hash of the pair table. Learned templates and edits
must be mined from its training parents only.

Run `python -m vp_metabolite.data verify` before training. This checks every
bundled source and processed table against `release/manifest.json`. The source
URLs and retrieval dates are in that manifest; source terms are in
`../LICENSE-DATA`.
