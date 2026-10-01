"""Enumerate products and describe their structures and generating routes."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from .chemistry import sanitised
from .edits import apply_edit, candidate_edits
from .verification import skeleton

FP_BITS = 256
SOURCES = ("published", "mined", "edit")


@dataclass
class Candidates:
    parent: str
    products: list[str]
    smiles: list[str]
    X: np.ndarray
    F: np.ndarray
    y: np.ndarray
    truth: set[str]


def _descriptors(mol) -> np.ndarray:
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return np.array(
        [
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
        ],
        dtype=float,
    )


def _count_fp(mol, fp_gen) -> np.ndarray:
    fp = fp_gen.GetCountFingerprint(mol)
    vec = np.zeros(FP_BITS, dtype=np.float32)
    for bit, count in fp.GetNonzeroElements().items():
        vec[bit % FP_BITS] += count
    return vec


def candidates_for(
    parent: str, published, mined, vocabulary, truth: set[str], fp_gen
) -> Candidates:
    from rdkit import Chem, DataStructs

    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return Candidates(
            parent,
            [],
            [],
            np.zeros((0, 26), np.float32),
            np.zeros((0, FP_BITS), np.float32),
            np.zeros(0),
            truth,
        )
    parent_smiles = Chem.MolToSmiles(mol, canonical=True)
    parent_desc = _descriptors(mol)
    parent_fp = fp_gen.GetFingerprint(mol)
    parent_counts = _count_fp(mol, fp_gen)
    routes: dict[str, list[tuple[float, int, int]]] = collections.defaultdict(list)
    best = {}

    def offer(clean, log_prior: float, phase: int, source: int) -> None:
        if clean is None or clean.GetNumHeavyAtoms() < 3:
            return
        smiles = Chem.MolToSmiles(clean, canonical=True)
        if smiles == parent_smiles:
            return
        key = skeleton(smiles)
        if not key:
            return
        routes[key].append((log_prior, phase, source))
        best.setdefault(key, clean)

    for log_prior, phase, rxn in published:
        try:
            sets = rxn.RunReactants((mol,))
        except Exception:
            continue
        for product_set in sets:
            for raw in product_set:
                offer(sanitised(raw), log_prior, phase, 0)
    for rxn in mined:
        try:
            sets = rxn.RunReactants((mol,))
        except Exception:
            continue
        for product_set in sets:
            for raw in product_set:
                offer(sanitised(raw), np.log(0.05), 1, 1)
    for kind, params, sites in candidate_edits(mol, vocabulary):
        try:
            smiles = apply_edit(mol, kind, params, sites)
        except Exception:
            continue
        if smiles is not None:
            offer(Chem.MolFromSmiles(smiles), np.log(0.05), 1, 2)

    products, product_smiles, rows, fps, labels = [], [], [], [], []
    for key in sorted(routes):
        entries = sorted(routes[key])
        product = best[key]
        priors = np.array([e[0] for e in entries])
        phases = np.array([e[1] for e in entries])
        srcs = np.array([e[2] for e in entries])
        desc = _descriptors(product)
        sim = DataStructs.TanimotoSimilarity(parent_fp, fp_gen.GetFingerprint(product))
        rows.append(
            np.concatenate(
                [
                    [
                        float(priors.max()),
                        float(np.log(np.exp(priors - priors.max()).sum()) + priors.max()),
                        float(len(entries)),
                        float(np.log1p(len(entries))),
                        float((phases == 1).sum()),
                        float((phases == 2).sum()),
                    ],
                    desc - parent_desc,
                    [float(sim)],
                    desc,
                    [float(parent_desc[0])],
                    [float((srcs == s).sum()) for s in range(len(SOURCES))],
                    [float((srcs == 0).any())],
                ]
            )
        )
        fps.append(_count_fp(product, fp_gen) - parent_counts)
        products.append(key)
        product_smiles.append(Chem.MolToSmiles(product, canonical=True))
        labels.append(float(key in truth))
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 26), np.float32)
    F = np.asarray(fps, dtype=np.float32) if fps else np.zeros((0, FP_BITS), np.float32)
    return Candidates(
        parent, products, product_smiles, X, F, np.asarray(labels, dtype=np.float32), truth
    )
