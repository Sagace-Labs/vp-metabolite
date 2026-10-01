from __future__ import annotations

MIN_PRODUCT_ATOMS = 3


def skeleton(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol).split("-")[0] if mol else None


def compile_template(smirks: str):
    from rdkit.Chem import AllChem

    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
    except Exception:
        return None
    if rxn is None or rxn.GetNumReactantTemplates() != 1:
        return None
    try:
        rxn.Initialize()
    except Exception:
        return None
    return rxn


def products_of(rxn, parent: str) -> set[str]:
    from rdkit import Chem

    from . import chemistry

    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return set()
    out: set[str] = set()
    try:
        sets = rxn.RunReactants((mol,))
    except Exception:
        return set()
    for product_set in sets:
        for raw in product_set:
            clean = chemistry.sanitised(raw)
            if clean is None or clean.GetNumHeavyAtoms() < MIN_PRODUCT_ATOMS:
                continue
            key = skeleton(Chem.MolToSmiles(clean, canonical=True))
            if key:
                out.add(key)
    return out


def verify(templates) -> tuple[list, dict]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    kept, stats = (
        [],
        {
            "templates": len(templates),
            "uncompilable": 0,
            "no_products": 0,
            "wrong_product": 0,
            "verified": 0,
        },
    )
    for tpl in templates:
        rxn = compile_template(tpl.smirks)
        if rxn is None:
            stats["uncompilable"] += 1
            continue
        parent, _, product = tpl.example.partition(">>")
        want = skeleton(product)
        made = products_of(rxn, parent)
        if not made:
            stats["no_products"] += 1
            continue
        if want not in made:
            stats["wrong_product"] += 1
            continue
        stats["verified"] += 1
        kept.append(tpl)
    return (kept, stats)
