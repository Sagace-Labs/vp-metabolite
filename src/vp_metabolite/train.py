"""Fit a new immutable metabolite version on the complete frozen pair table."""

from __future__ import annotations

import argparse
import platform
from datetime import date
from pathlib import Path

import joblib
import rdkit
import sklearn

import vp_core
from vp_core import manifest, protocols
from vp_core.hashing import sha256_file

from . import PATHWAY, VERSIONS_DIR, contract, data, model

PROTOCOL_ID = "metabolite-v1-fixed-similarity@1"


def build_version(version: str, reason: str, *, cache_dir: Path | None = None) -> None:
    destination = VERSIONS_DIR / version
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; released versions are immutable")
    protocol = protocols.get(PROTOCOL_ID)
    release = data.verify()
    rows = data.load_pairs()
    pairs = data.training_rows(rows)
    print(f"mining from {len(pairs)} directed pairs", flush=True)
    smirks, edits = model.fit_generator(pairs)
    print(f"verified templates: {len(smirks)}; edit types: {len(edits)}", flush=True)
    truth = {}
    parents = {}
    for row in rows:
        truth.setdefault(row["parent_key"], set()).add(row["metabolite_key"])
        parents.setdefault(row["parent_key"], row["parent_smiles"])
    tasks = [(parents[key], truth[key]) for key in sorted(parents)]
    forest = model.fit_ranker_stream(tasks, smirks, edits, seed=0, cache_dir=cache_dir)
    destination.mkdir(parents=True)
    weights = destination / "weights.joblib"
    joblib.dump({"forest": forest, "smirks": smirks, "edits": edits}, weights, compress=3)
    record = {
        "schema": manifest.SCHEMA_VERSION,
        "pathway": PATHWAY,
        "version": version,
        "released": date.today().isoformat(),
        "reason": reason,
        "signature": contract.as_manifest_table(),
        "dataset": {
            "name": "Five-source xenobiotic parent-metabolite relationships",
            "source": "MetXBioDB, ChEMBL metabolism, e-Drug3D, observed MetSim and NCATS Inxight/GSRS",
            "retrieved": "2026-09-21",
            "licence": "CC-BY-SA-4.0",
            "redistributable": True,
            "path": "data/release/pairs.csv",
            "format": "directed-pairs@1",
            "unit": "directed pairs",
            "sha256": release["files"]["pairs.csv"]["sha256"],
            "n_rows": len(rows),
            "fetch": "python -m vp_metabolite.data verify",
        },
        "model": {
            "family": "verified-rule and graph-edit generator with random-forest ranker",
            "features": model.FEATURES,
            "fit": f"all {len(parents)} parents; 1,200 train-pair radius-one template sample and train-only edit vocabulary",
            "weights": weights.name,
            "sha256": sha256_file(weights),
        },
        "protocol": {
            "id": protocol.id,
            "provider": "vp-core",
            "core_version": vp_core.__version__,
        },
        "provenance": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    manifest.write(destination / "manifest.toml", record)
    problems = manifest.validate(record, version_dir=destination)
    if problems:
        raise ValueError("; ".join(problems))
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    build_version(args.version, args.reason, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
