"""Exercise the bundled chemistry and released prediction contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from vp_metabolite import contract, data, get, model, predict
from vp_metabolite.rules import compiled_rules


def test_rules_compile_and_fixture_generates_products():
    fixture = pd.read_parquet(Path(__file__).parents[1] / "data/example/pairs_example.parquet")
    assert len(compiled_rules()) == 467
    row = fixture.iloc[0]
    example = model.generate(row.parent_smiles, [], [], {row.metabolite_key})
    assert example.X.shape == (len(example.products), 26)
    assert example.F.shape == (len(example.products), 256)
    assert len(set(example.products)) == len(example.products)
    assert example.products == sorted(example.products)
    assert len(example.smiles) == len(example.products)


def test_release_table_and_split_hashes():
    release = data.verify()
    split = Path(__file__).parents[1] / "data/confirmation_split.json"
    from vp_core.protocols import get as get_protocol

    protocol = get_protocol("metabolite-v1-fixed-similarity@1")
    assert hashlib.sha256(split.read_bytes()).hexdigest() == protocol.split_key_sha256
    assert release["report"]["unique_pairs"] == 7081


def test_prediction_uses_version_frame():
    released = get("v1")
    frame = predict(["CC(=O)Nc1ccc(O)cc1", "not a smiles"], limit=3)
    assert list(frame.columns) == [contract.OUTPUTS[0]["name"]]
    assert list(frame.columns) == released.output_names
    assert frame.index.names == [
        "input_index",
        "rank",
        "metabolite_smiles",
        "connectivity_key",
    ]
    assert len(frame) <= 3
    assert set(frame.index.get_level_values("input_index")) <= {0}
