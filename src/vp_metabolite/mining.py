from __future__ import annotations

import collections
from dataclasses import dataclass

MIN_MCS_FRACTION = 0.55
MCS_TIMEOUT = 5


@dataclass(frozen=True)
class MinedTemplate:
    smirks: str
    radius: int
    support: int
    example: str
    centre_size: int

    @property
    def name(self) -> str:
        return f"mined_r{self.radius}_{abs(hash(self.smirks)) % 10**8:08d}"


def _atom_signature(atom) -> tuple:
    return (
        atom.GetAtomicNum(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        atom.GetDegree(),
        atom.GetIsAromatic(),
        tuple(sorted(b.GetBondTypeAsDouble() for b in atom.GetBonds())),
    )


def map_atoms(parent, product):
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    res = rdFMCS.FindMCS(
        [parent, product],
        timeout=MCS_TIMEOUT,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
    )
    if res.canceled or res.numAtoms < 3:
        return None
    if res.numAtoms < MIN_MCS_FRACTION * parent.GetNumAtoms():
        return None
    core = Chem.MolFromSmarts(res.smartsString)
    if core is None:
        return None
    pm = parent.GetSubstructMatch(core)
    qm = product.GetSubstructMatch(core)
    if not pm or not qm or len(pm) != len(qm):
        return None
    return dict(zip(pm, qm, strict=True))


def reaction_centre(parent, product, mapping: dict[int, int]) -> set[int]:
    centre = {i for i in range(parent.GetNumAtoms()) if i not in mapping}
    for pi, qi in mapping.items():
        if _atom_signature(parent.GetAtomWithIdx(pi)) != _atom_signature(
            product.GetAtomWithIdx(qi)
        ):
            centre.add(pi)
    for pi in list(centre):
        for nb in parent.GetAtomWithIdx(pi).GetNeighbors():
            if nb.GetIdx() not in mapping:
                centre.add(nb.GetIdx())
    return centre


def _expand(mol, seed: set[int], radius: int) -> set[int]:
    out = set(seed)
    frontier = set(seed)
    for _ in range(radius):
        nxt: set[int] = set()
        for i in frontier:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() not in out:
                    nxt.add(nb.GetIdx())
        out |= nxt
        frontier = nxt
    return out


def template_smarts(
    parent, product, mapping: dict[int, int], centre: set[int], radius: int
) -> str | None:
    from rdkit import Chem

    keep_p = _expand(parent, centre, radius)
    if not keep_p or len(keep_p) > 40:
        return None
    keep_q = {mapping[i] for i in keep_p if i in mapping}
    added = set(range(product.GetNumAtoms())) - set(mapping.values())
    if added:
        anchor = {mapping[i] for i in keep_p if i in mapping}
        keep_q |= {j for j in _expand(product, anchor, radius + 1) if j in added}
    if not keep_q or len(keep_q) > 60:
        return None
    p_labels = {pi: n + 1 for n, pi in enumerate(sorted(keep_p))}
    q_labels = {mapping[pi]: lab for pi, lab in p_labels.items() if pi in mapping}

    def fragment(mol, keep, labels):
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            atom.SetAtomMapNum(labels.get(atom.GetIdx(), 0))
        for i in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
            rw.RemoveAtom(i)
        frag = rw.GetMol()
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            return None
        return Chem.MolToSmarts(frag)

    left = fragment(parent, keep_p, p_labels)
    right = fragment(product, keep_q, q_labels)
    if not left or not right or left == right:
        return None
    return f"{left}>>{right}"


def mine_templates(
    pairs, *, radius: int = 1, limit: int | None = None, min_support: int = 1
) -> tuple[list[MinedTemplate], dict]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    counts: collections.Counter = collections.Counter()
    example: dict[str, str] = {}
    centre_of: dict[str, int] = {}
    stats = {
        "pairs": 0,
        "unparsed": 0,
        "unmapped": 0,
        "no_centre": 0,
        "no_template": 0,
        "extracted": 0,
    }
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
        centre = reaction_centre(parent, product, mapping)
        if not centre or len(centre) > 12:
            stats["no_centre"] += 1
            continue
        smirks = template_smarts(parent, product, mapping, centre, radius)
        if smirks is None:
            stats["no_template"] += 1
            continue
        counts[smirks] += 1
        stats["extracted"] += 1
        example.setdefault(smirks, f"{pair.parent}>>{pair.metabolite}")
        centre_of.setdefault(smirks, len(centre))
    mined = [
        MinedTemplate(s, radius, c, example[s], centre_of[s])
        for s, c in counts.items()
        if c >= min_support
    ]
    mined.sort(key=lambda t: -t.support)
    stats["unique"] = len(counts)
    stats["kept"] = len(mined)
    return (mined, stats)
