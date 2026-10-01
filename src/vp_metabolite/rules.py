"""Load the bundled transformation rules and report skipped entries."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .verification import compile_template

RULES_DIR = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class Rule:
    name: str
    smirks: str
    source: str
    phase: int
    prior: float


def load_rules() -> list[Rule]:
    out = []
    for row in json.loads((RULES_DIR / "curated.json").read_text(encoding="utf-8")):
        out.append(Rule(row["name"], row["smirks"], "curated", row["phase"], row["prior"]))
    for phase in (1, 2):
        path = RULES_DIR / f"sygma_phase{phase}.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            try:
                prior = float(parts[1])
            except ValueError:
                continue
            out.append(Rule(parts[2].strip(), parts[0].strip(), "sygma", phase, prior))
    with (RULES_DIR / "gloryx_rules.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            smirks = (row.get("SMIRKS") or "").strip()
            if not smirks:
                continue
            phase = 2 if "phase 2" in (row.get("Name of rule subset") or "").lower() else 1
            prior = 1.0 if row.get("Priority level") == "common" else 0.3
            out.append(Rule(row.get("Reaction name") or "", smirks, "gloryx", phase, prior))
    return out


@lru_cache(maxsize=1)
def compiled_rules() -> tuple[tuple[float, int, object], ...]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for rule in load_rules():
        rxn = compile_template(rule.smirks)
        if rxn is not None:
            out.append((math.log(max(float(rule.prior or 1.0), 1e-6)), rule.phase, rxn))
    return tuple(out)
