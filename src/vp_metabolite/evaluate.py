"""Evaluate a release recipe on the frozen similarity-separated parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from vp_core import card, metrics, metrics_store, protocols

from . import PATHWAY, data, get, model
from .train import PROTOCOL_ID

SPLIT = Path(__file__).resolve().parents[2] / "data" / "confirmation_split.json"


def _measure_held(held_tasks, smirks, edits, forest, cache_dir) -> dict[str, float]:
    scores = {k: [] for k in (1, 5, 10, 20)}
    reachable, counts, prior_scores = [], [], []
    examples = model.iter_generated(
        held_tasks, smirks, edits, label="test", cache_dir=cache_dir
    )
    for i, example in enumerate(examples, 1):
        truth = example.truth
        if not truth:
            continue
        order = model.rank(forest, example)
        counts.append(len(order))
        for k in scores:
            scores[k].append(
                len({product for product, _, _ in order[:k]} & truth) / len(truth)
            )
        reachable.append(len(set(example.products) & truth) / len(truth))
        prior = np.argsort(-example.X[:, 0], kind="stable")[:10]
        prior_scores.append(len({example.products[j] for j in prior} & truth) / len(truth))
        if i % 100 == 0:
            print(f"scored test {i}/{len(held_tasks)} parents", flush=True)
    return {
        **{f"recall_at_{k}": float(np.mean(values)) for k, values in scores.items()},
        "reachable_recall": float(np.mean(reachable)),
        "candidates_per_parent": float(np.mean(counts)),
        "prior_recall_at_10": float(np.mean(prior_scores)),
    }


def _evaluate_recipe(
    version: str, expected: str, dataset_name: str, cache_dir: Path | None
) -> dict:
    protocol = protocols.get(PROTOCOL_ID)
    rows = data.load_pairs()
    if data.release_info()["files"]["pairs.csv"]["sha256"] != expected:
        raise ValueError("release pair table differs from the version manifest")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != protocol.split_key_sha256:
        raise ValueError("evaluation split differs from the registered protocol")
    if split["pairs_sha256"] != expected:
        raise ValueError("evaluation split belongs to another pair table")
    train_keys = set(split["train_parent_keys"])
    test_keys = set(split["test_parent_keys"])
    if train_keys & test_keys:
        raise ValueError("train and test parent identities overlap")
    present = {row["parent_key"] for row in rows}
    if not train_keys | test_keys <= present:
        raise ValueError("split contains a parent missing from the pair table")
    parents = {}
    truth = {}
    for row in rows:
        parents.setdefault(row["parent_key"], row["parent_smiles"])
        truth.setdefault(row["parent_key"], set()).add(row["metabolite_key"])
    pairs = data.training_rows(rows, train_keys)
    print(f"mining from {len(pairs)} held-in directed pairs", flush=True)
    smirks, edits = model.fit_generator(pairs)
    print(f"verified templates: {len(smirks)}; edit types: {len(edits)}", flush=True)
    training_tasks = [(parents[key], truth[key]) for key in sorted(train_keys)]
    held_tasks = [(parents[key], truth[key]) for key in sorted(test_keys)]
    forest = model.fit_ranker_stream(
        training_tasks, smirks, edits, seed=protocol.seeds[0], cache_dir=cache_dir
    )
    per_seed = []
    folds = []
    for seed in protocol.seeds:
        if seed != protocol.seeds[0]:
            forest = model.fit_ranker_stream(
                training_tasks, smirks, edits, seed=seed, cache_dir=cache_dir
            )
        measured = _measure_held(held_tasks, smirks, edits, forest, cache_dir)
        per_seed.append(measured)
        folds.append(
            {
                "seed": seed,
                "n_train": len(training_tasks),
                "n_val": 0,
                "n_test": len(held_tasks),
                "mined_templates": len(smirks),
                "edit_types": len(edits),
            }
        )
    entry = {
        "pathway": PATHWAY,
        "version": version,
        "protocol_id": protocol.id,
        "dataset_sha256": expected,
        "dataset": dataset_name,
        "evaluated": date.today().isoformat(),
        "n": {"train": len(training_tasks), "val": 0, "test": len(held_tasks)},
        "folds": folds,
        "test": metrics.aggregate(per_seed, protocol.metrics),
    }
    return entry


def evaluate_version(version: str, *, cache_dir: Path | None = None) -> dict:
    released = get(version)
    entry = _evaluate_recipe(
        version,
        released.manifest["dataset"]["sha256"],
        released.manifest["dataset"]["name"],
        cache_dir,
    )
    metrics_store.write(released.directory, entry)
    card.write_card(released.directory)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(evaluate_version(args.version, cache_dir=args.cache_dir)["test"], indent=2)
    )


if __name__ == "__main__":
    main()
