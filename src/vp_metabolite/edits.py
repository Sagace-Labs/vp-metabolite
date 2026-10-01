from __future__ import annotations

from dataclasses import dataclass

ATTACH = "attach"
BOND = "bond"
CLEAVE = "cleave"
MAX_FRAGMENT_ATOMS = 24


@dataclass(frozen=True)
class Edit:
    kind: str
    params: tuple
    sites: tuple[int, ...]

    @property
    def type_key(self) -> tuple:
        return (self.kind, *self.params)

    def __str__(self) -> str:
        return f"{self.kind}{self.params}@{self.sites}"


def _fragment_smiles(mol, atoms: set[int], anchor: int) -> tuple[str, float] | None:
    from rdkit import Chem

    if not atoms or len(atoms) > MAX_FRAGMENT_ATOMS:
        return None
    contact = [i for i in atoms if mol.GetBondBetweenAtoms(i, anchor) is not None]
    if len(contact) != 1:
        return None
    rw = Chem.RWMol(mol)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(dummy, contact[0], Chem.BondType.SINGLE)
    keep = atoms | {dummy}
    for i in sorted(set(range(rw.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(i)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    order = mol.GetBondBetweenAtoms(contact[0], anchor).GetBondTypeAsDouble()
    return (Chem.MolToSmiles(frag, canonical=True), order)


def _components(mol, atoms: set[int]) -> list[set[int]]:
    remaining = set(atoms)
    out = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining:
                    remaining.discard(j)
                    group.add(j)
                    stack.append(j)
        out.append(group)
    return out


def _order(mol, i: int, j: int) -> float:
    bond = mol.GetBondBetweenAtoms(i, j)
    return bond.GetBondTypeAsDouble() if bond else 0.0


def extract_edit(parent, product, mapping: dict[int, int]) -> Edit | None:
    inverse = {q: p for p, q in mapping.items()}
    added = set(range(product.GetNumAtoms())) - set(inverse)
    removed = set(range(parent.GetNumAtoms())) - set(mapping)
    changed_bonds = []
    mapped = sorted(mapping)
    for a in range(len(mapped)):
        for b in range(a + 1, len(mapped)):
            i, j = (mapped[a], mapped[b])
            po, qo = (_order(parent, i, j), _order(product, mapping[i], mapping[j]))
            if po != qo:
                changed_bonds.append((i, j, po, qo))
    if added and (not removed) and (not changed_bonds):
        pieces = _components(product, added)
        if len(pieces) != 1:
            return None
        piece = pieces[0]
        anchors = {
            inverse[nb.GetIdx()]
            for i in piece
            for nb in product.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() in inverse
        }
        if len(anchors) != 1:
            return None
        site = anchors.pop()
        got = _fragment_smiles(product, piece, mapping[site])
        if got is None:
            return None
        frag, order = got
        dq = (
            product.GetAtomWithIdx(mapping[site]).GetFormalCharge()
            - parent.GetAtomWithIdx(site).GetFormalCharge()
        )
        return Edit(ATTACH, (frag, dq, order), (site,))
    if not added and (not removed) and (len(changed_bonds) == 1):
        i, j, po, qo = changed_bonds[0]
        if po == 0.0 or qo == 0.0:
            return None
        return Edit(BOND, (po, qo), (i, j))
    if removed and len(changed_bonds) <= 1:
        gone = _components(parent, removed)
        if len(gone) != 1:
            return None
        piece = gone[0]
        border = [
            (nb.GetIdx(), i)
            for i in piece
            for nb in parent.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() in mapping
        ]
        if len(border) != 1:
            return None
        site, partner = border[0]
        if added:
            got = _fragment_smiles(product, added, mapping[site])
            if got is None:
                return None
            kept, arriving = got
        else:
            kept, arriving = ("[H]", 0.0)
        broken = _order(parent, site, partner)
        return Edit(CLEAVE, (kept, arriving, broken), (site, partner))
    return None


def _finalise(rw, parent_smiles: str, keep_atom: int | None = None):
    from rdkit import Chem

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    pieces = Chem.GetMolFrags(mol)
    if len(pieces) > 1:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if keep_atom is not None:
            chosen = next(
                (f for f, idx in zip(frags, pieces, strict=True) if keep_atom in idx), None
            )
            if chosen is None:
                return None
            mol = chosen
        else:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() < 3:
        return None
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return None if smiles == parent_smiles else smiles


def apply_edit(mol, kind: str, params: tuple, sites: tuple[int, ...]) -> str | None:
    from rdkit import Chem

    parent_smiles = Chem.MolToSmiles(mol, canonical=True)
    if kind == ATTACH:
        frag_smiles, dq, order = params
        (site,) = sites
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return None
        atom = mol.GetAtomWithIdx(site)
        if atom.GetTotalNumHs() < 1 and dq == 0 and (order < 2):
            return None
        rw = Chem.RWMol(mol)
        offset = rw.GetNumAtoms()
        rw.InsertMol(frag)
        dummy = next(
            (
                a.GetIdx()
                for a in rw.GetAtoms()
                if a.GetIdx() >= offset and a.GetAtomicNum() == 0
            ),
            None,
        )
        if dummy is None:
            return None
        neighbours = [nb.GetIdx() for nb in rw.GetAtomWithIdx(dummy).GetNeighbors()]
        if len(neighbours) != 1:
            return None
        rw.AddBond(site, neighbours[0], _BOND_TYPES[order])
        rw.RemoveAtom(dummy)
        target = rw.GetAtomWithIdx(site)
        target.SetNoImplicit(True)
        target.SetNumExplicitHs(max(atom.GetTotalNumHs() - int(order), 0))
        target.SetFormalCharge(atom.GetFormalCharge() + dq)
        return _finalise(rw, parent_smiles)
    if kind == BOND:
        want, become = params
        i, j = sites
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetBondTypeAsDouble() != want:
            return None
        rw = Chem.RWMol(mol)
        rw.RemoveBond(i, j)
        rw.AddBond(i, j, _BOND_TYPES[become])
        for idx in (i, j):
            a = rw.GetAtomWithIdx(idx)
            spare = mol.GetAtomWithIdx(idx).GetTotalNumHs()
            need = int(become - want)
            if need > 0:
                if spare < need:
                    return None
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(spare - need)
        return _finalise(rw, parent_smiles)
    if kind == CLEAVE:
        kept, arriving, broken = params
        site, partner = sites
        bond = mol.GetBondBetweenAtoms(site, partner)
        if bond is None or bond.IsInRing():
            return None
        if bond.GetBondTypeAsDouble() != broken:
            return None
        spare = mol.GetAtomWithIdx(site).GetTotalNumHs() + int(broken) - int(arriving)
        if spare < 0:
            return None
        rw = Chem.RWMol(mol)
        rw.RemoveBond(site, partner)
        atom = rw.GetAtomWithIdx(site)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(spare)
        if kept != "[H]":
            frag = Chem.MolFromSmiles(kept)
            if frag is None:
                return None
            offset = rw.GetNumAtoms()
            rw.InsertMol(frag)
            dummy = next(
                (
                    a.GetIdx()
                    for a in rw.GetAtoms()
                    if a.GetIdx() >= offset and a.GetAtomicNum() == 0
                ),
                None,
            )
            if dummy is None:
                return None
            nbs = [nb.GetIdx() for nb in rw.GetAtomWithIdx(dummy).GetNeighbors()]
            if len(nbs) != 1:
                return None
            rw.AddBond(site, nbs[0], _BOND_TYPES[arriving])
            rw.RemoveAtom(dummy)
        return _finalise(rw, parent_smiles, keep_atom=site)
    return None


def candidate_edits(mol, vocabulary) -> list[tuple[str, tuple, tuple[int, ...]]]:
    out = []
    n = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    for kind, *params in vocabulary:
        params = tuple(params)
        if kind == ATTACH:
            out.extend((kind, params, (i,)) for i in range(n))
        elif kind == BOND:
            out.extend(((kind, params, (i, j)) for i, j in bonds))
        elif kind == CLEAVE:
            for i, j in bonds:
                out.append((kind, params, (i, j)))
                out.append((kind, params, (j, i)))
    return out


def _bond_types():
    from rdkit import Chem

    return {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
    }


class _LazyBondTypes(dict):
    def __missing__(self, key):
        self.update(_bond_types())
        return self[key]


_BOND_TYPES = _LazyBondTypes()
