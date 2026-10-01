"""Fit the bounded candidate generator and its product ranker."""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import pickle
import random
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .generator import candidates_for
from .mining import mine_templates
from .rules import compiled_rules
from .verification import compile_template, verify
from .vocabulary import decompose

MINING_PAIRS = 1200
MIN_EDIT_SUPPORT = 5
FEATURES = (
    "route counts, product descriptors, parent-product deltas and difference fingerprint"
)
FEATURES_BY_OUTPUT = {}
_WORKER_GENERATOR = None


def features_for(output: str) -> str:
    return FEATURES_BY_OUTPUT.get(output, FEATURES)


def fit_generator(pairs, *, seed: int = 95) -> tuple[list[str], list[tuple]]:
    pairs = sorted(pairs, key=lambda p: (p.parent, p.metabolite))
    random.Random(seed).shuffle(pairs)
    mined, _ = mine_templates(pairs[:MINING_PAIRS], radius=1)
    verified, _ = verify(mined)
    vocabulary, _, _, _ = decompose(pairs, verify=False)
    edits = [(key, count) for key, count in vocabulary.items() if count >= MIN_EDIT_SUPPORT]
    edits.sort(key=lambda item: (-item[1], repr(item[0])))
    return sorted(item.smirks for item in verified), [key for key, _ in edits]


@lru_cache(maxsize=4)
def compile_generator(smirks: tuple[str, ...]):
    return [rxn for rule in smirks if (rxn := compile_template(rule)) is not None]


def generate(
    parent: str, smirks: list[str], edits: list[tuple], truth: set[str] | None = None
):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return candidates_for(
        parent,
        compiled_rules(),
        compile_generator(tuple(smirks)),
        edits,
        truth or set(),
        fp_gen,
    )


def _init_worker(smirks, edits):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = smirks, edits


def _generate_worker(task):
    parent, truth = task
    smirks, edits = _WORKER_GENERATOR
    return generate(parent, smirks, edits, truth)


def iter_generated(tasks, smirks, edits, *, label: str, cache_dir: Path | None = None):
    """Yield parents in order; optional local cache resumes interrupted runs."""
    workers = max(
        1, min(int(os.environ.get("VP_METABOLITE_WORKERS", "2")), os.cpu_count() or 1)
    )
    if cache_dir is not None:
        package = Path(__file__).parent
        source_hash = hashlib.sha256()
        source_files = [
            package / name
            for name in (
                "model.py",
                "generator.py",
                "rules.py",
                "chemistry.py",
                "edits.py",
                "verification.py",
            )
        ]
        source_files.extend(sorted((package / "resources").glob("*")))
        for source in source_files:
            source_hash.update(source.name.encode())
            source_hash.update(source.read_bytes())
        identity = json.dumps(
            [
                source_hash.hexdigest(),
                smirks,
                edits,
                [(parent, sorted(truth)) for parent, truth in tasks],
            ],
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        cache_dir = cache_dir / f"{label}-{digest}"
        cache_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for i, task in enumerate(tasks):
        cached = cache_dir / f"{i:05d}.pkl.gz" if cache_dir is not None else None
        if cached is None or not cached.exists():
            pending.append((i, task))
    print(
        f"{label}: {len(tasks) - len(pending)} cached, {len(pending)} to generate", flush=True
    )
    if not pending:
        for i in range(len(tasks)):
            with gzip.open(cache_dir / f"{i:05d}.pkl.gz", "rb") as stream:
                yield pickle.load(stream)
        return
    with get_context("spawn").Pool(
        processes=workers, initializer=_init_worker, initargs=(smirks, edits)
    ) as pool:
        computed = pool.imap(_generate_worker, (task for _, task in pending), chunksize=1)
        completed = 0
        for index in range(len(tasks)):
            cached = cache_dir / f"{index:05d}.pkl.gz" if cache_dir is not None else None
            if cached is not None and cached.exists():
                with gzip.open(cached, "rb") as stream:
                    example = pickle.load(stream)
            else:
                example = next(computed)
                completed += 1
                if cached is not None:
                    temporary = cached.with_suffix(".tmp")
                    with gzip.open(temporary, "wb", compresslevel=1) as stream:
                        pickle.dump(example, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    temporary.replace(cached)
                if completed % 25 == 0:
                    print(f"generated {label} {completed}/{len(pending)} parents", flush=True)
            yield example


def generate_batch(tasks, smirks, edits, *, label: str, cache_dir: Path | None = None):
    return list(iter_generated(tasks, smirks, edits, label=label, cache_dir=cache_dir))


def fit_ranker(examples, *, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    xs, ys = [], []
    for example in examples:
        if len(example.products) and example.truth:
            xs.append(np.concatenate([example.X, example.F], axis=1))
            ys.append(example.y)
    X, y = np.concatenate(xs), np.concatenate(ys)
    fitted = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=max(1, int(os.environ.get("VP_METABOLITE_JOBS", "2"))),
    )
    fitted.fit(X, y)
    return fitted


def fit_ranker_stream(tasks, smirks, edits, *, seed: int, cache_dir: Path | None = None):
    """Fit the same forest from a file-backed matrix to bound RAM use."""
    from sklearn.ensemble import RandomForestClassifier

    with TemporaryDirectory(prefix="vp-metabolite-fit-") as temporary:
        work = Path(temporary)
        cache = cache_dir if cache_dir is not None else work / "candidates"
        count = 0
        columns = None
        for example in iter_generated(tasks, smirks, edits, label="train", cache_dir=cache):
            if example.products and example.truth:
                count += len(example.products)
                columns = example.X.shape[1] + example.F.shape[1]
        if not count or columns is None:
            raise ValueError("no labelled candidate rows for ranker fit")
        print(f"fitting forest on {count} candidate rows", flush=True)
        X = np.memmap(
            work / "features.dat",
            dtype=np.float32,
            mode="w+",
            shape=(count, columns),
            order="F",
        )
        y = np.memmap(work / "labels.dat", dtype=np.uint8, mode="w+", shape=count)
        cursor = 0
        for example in iter_generated(tasks, smirks, edits, label="train", cache_dir=cache):
            if example.products and example.truth:
                size = len(example.products)
                X[cursor : cursor + size] = np.concatenate([example.X, example.F], axis=1)
                y[cursor : cursor + size] = example.y
                cursor += size
        X.flush()
        y.flush()
        fitted = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=max(1, int(os.environ.get("VP_METABOLITE_JOBS", "2"))),
        )
        fitted.fit(X, y)
        del X, y
        gc.collect()
        return fitted


def rank(fitted, example):
    if not example.products:
        return []
    design = np.concatenate([example.X, example.F], axis=1)
    scores = fitted.predict_proba(design)[:, 1]
    order = np.argsort(-scores, kind="stable")
    return [(example.products[i], example.smiles[i], float(scores[i])) for i in order]
