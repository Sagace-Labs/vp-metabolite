from __future__ import annotations


def sanitised(mol):
    from rdkit import Chem

    base = Chem.RWMol(mol)
    for atom in base.GetAtoms():
        if atom.HasProp("old_mapno"):
            atom.SetNoImplicit(False)
    direct = base.GetMol()
    try:
        Chem.SanitizeMol(direct)
        return direct
    except Exception:
        pass
    candidates = [
        a.GetIdx()
        for a in base.GetAtoms()
        if a.GetIsAromatic()
        and a.GetAtomicNum() == 7
        and (a.GetDegree() == 2)
        and (a.GetFormalCharge() == 0)
        and (a.GetTotalNumHs() == 0)
    ]
    for idx in candidates:
        trial = Chem.RWMol(base)
        atom = trial.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        out = trial.GetMol()
        try:
            Chem.SanitizeMol(out)
            return out
        except Exception:
            continue
    return None
