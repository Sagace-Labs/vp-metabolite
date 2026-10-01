"""Load the frozen directed-pair release and verify its recorded hashes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

DATA = Path(__file__).resolve().parents[2] / "data" / "release"


def release_info() -> dict:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def verify() -> dict:
    record = release_info()
    for relative, expected in record["files"].items():
        path = DATA / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"release file hash mismatch: {relative}")
    return record


def load_pairs() -> list[dict[str, str]]:
    verify()
    with (DATA / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def training_rows(rows: list[dict[str, str]], keys: set[str] | None = None):
    selected = rows if keys is None else [row for row in rows if row["parent_key"] in keys]
    return [
        SimpleNamespace(
            parent=row["parent_smiles"],
            metabolite=row["metabolite_smiles"],
            sources=row["sources"],
        )
        for row in selected
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    args = parser.parse_args()
    if args.command == "verify":
        record = verify()
        print(f"verified {record['report']['unique_pairs']} directed pairs")


if __name__ == "__main__":
    main()
