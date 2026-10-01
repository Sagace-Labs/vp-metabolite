"""Rank plausible xenobiotic metabolite structures from parent SMILES."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vp_core.registry import Version, VersionedPathway

from . import model

__version__ = "1.0.0"
PATHWAY = "metabolite"
VERSIONS_DIR = Path(__file__).resolve().parent / "versions"
_pathway = VersionedPathway(PATHWAY, VERSIONS_DIR)


def versions() -> list[str]:
    return _pathway.versions()


def current_version() -> str:
    return _pathway.current()


def get(version: str | None = None) -> Version:
    return _pathway.get(version)


def signature(version: str | None = None) -> dict[str, Any]:
    return get(version).manifest["signature"]


def predict(smiles: list[str], *, version: str | None = None, limit: int = 10) -> pd.DataFrame:
    """Return a ranked frame indexed by input, rank, structure and connectivity."""
    if isinstance(smiles, str):
        raise TypeError("pass a list of SMILES, not a single string")
    if limit < 1:
        raise ValueError("limit must be positive")
    released = get(version)
    fitted = released.load_model()
    rows = []
    values = []
    for input_index, parent in enumerate(smiles):
        example = model.generate(parent, fitted["smirks"], fitted["edits"])
        for rank, (key, product, score) in enumerate(
            model.rank(fitted["forest"], example)[:limit], 1
        ):
            rows.append((input_index, rank, product, key))
            values.append(score)
    frame = released.frame(np.asarray(values, dtype=np.float32).reshape(-1, 1))
    frame.index = pd.MultiIndex.from_tuples(
        rows, names=("input_index", "rank", "metabolite_smiles", "connectivity_key")
    )
    return frame
