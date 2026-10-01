from __future__ import annotations

import collections

from .edits import Edit, apply_edit, extract_edit
from .mining import map_atoms


def _skeleton(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol).split("-")[0] if mol else None


def decompose(pairs, *, limit: int | None = None, verify: bool = True):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    stats = collections.Counter()
    vocabulary: collections.Counter = collections.Counter()
    example: dict[tuple, str] = {}
    centre: dict[tuple, list[int]] = collections.defaultdict(list)
    extracted: list[tuple] = []
    for n, pair in enumerate(pairs):
        if limit is not None and n >= limit:
            break
        stats["pairs"] += 1
        parent = Chem.MolFromSmiles(pair.parent)
        product = Chem.MolFromSmiles(pair.metabolite)
        if parent is None or product is None:
            stats["unparsed"] += 1
            continue
        mapping = map_atoms(parent, product)
        if mapping is None:
            stats["unmapped"] += 1
            continue
        edit: Edit | None = extract_edit(parent, product, mapping)
        if edit is None:
            stats["not_one_edit"] += 1
            continue
        if verify:
            got = apply_edit(parent, edit.kind, edit.params, edit.sites)
            if got is None or _skeleton(got) != _skeleton(
                Chem.MolToSmiles(product, canonical=True)
            ):
                stats["failed_roundtrip"] += 1
                continue
            if got != Chem.MolToSmiles(product, canonical=True):
                stats["stereo_only"] += 1
        stats["extracted"] += 1
        key = edit.type_key
        vocabulary[key] += 1
        example.setdefault(key, f"{pair.parent}>>{pair.metabolite}")
        centre[key].append(len(edit.sites))
        extracted.append((pair.parent, pair.metabolite, key, edit.sites))
    return (vocabulary, stats, example, extracted)
