"""Load frozen 256-dimensional projections and calibrate learned HNSW search."""
from functools import partial
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .benchmark import create_index, evaluate
from .calibration import choose, measure_p95, transform
from .io import fingerprint, read_json, run_directory, write_json, write_rows
from .phase3 import configuration, verified_file
from .projection import development_data, load_projection


def learned256(lock, training):
    path = Path(lock["training_summary"]["file"]).parent / "learned256.npz"
    record = dict(file=str(path), sha256=training["weights_sha256"][path.name])
    mean, weights = load_projection(verified_file(record))
    if weights.shape != (256, 768):
        raise ValueError("Unexpected learned256 weight dimensions")
    return partial(transform, mean=mean, components=weights), record


def pca256(lock):
    path = verified_file(lock["pca"])
    mean, weights = load_projection(path)
    if weights.shape != (256, 768):
        raise ValueError("Unexpected PCA weight dimensions")
    return partial(transform, mean=mean, components=weights)


def calibrate(args):
    args.passages = 100000
    lock, cache, _ = configuration(args)
    training = read_json(verified_file(lock["training_summary"]))
    for name in ("corpus-dev.jsonl", "eval-dev.jsonl"):
        if fingerprint(args.prepared / name) != training["inputs"][name]:
            raise ValueError(f"Development input differs from training: {name}")
    preprocess, checkpoint = learned256(lock, training)
    reference = lock["sizes"][str(args.passages)]["flat_k10"]["raw768"]
    grid = lock["formal_test_grid"]
    seed = lock["index_seed"]
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        faiss.omp_set_num_threads(lock["build_threads"])
        ids, docs, dev, queries = development_data(args.prepared, args.passages, cache)
        indexed, q = preprocess(docs), preprocess(queries)
        del docs
        index, name, _ = create_index(indexed, "learned256", out, seed, grid["primary_m"])
        faiss.omp_set_num_threads(lock["search_threads"])
        candidates = []
        for ef in grid["ef_search"]["10"]:
            index.hnsw.efSearch = ef
            rows, _, quality = evaluate(index, ids, [("dev", dev)], q, 10, None)
            write_rows(out / f"{name}-ef{ef}.jsonl", rows)
            candidates.append(dict(representation="learned256", ef_search=ef, quality=quality["dev"],
                                   m=grid["primary_m"], k=10, repeated_p95_ms=[]))
        rng = np.random.default_rng(seed)
        for _ in range(5):
            order = rng.permutation(len(dev))
            for i in rng.permutation(len(candidates)):
                point = candidates[i]
                index.hnsw.efSearch = point["ef_search"]
                point["repeated_p95_ms"].append(measure_p95(index, preprocess, queries, ids, 10, order))
        for p in candidates:
            p["p95_ms"] = float(np.median(p["repeated_p95_ms"]))
        selected = choose([dict(reference, representation="raw768"), *candidates], 10, ["learned256"])
        write_json(out / "lock.json", dict(selected=selected["learned256"], configurations=candidates,
                   passages=len(ids), queries=len(dev), seed=seed, reference=reference, split="MS MARCO dev",
                   source_lock_sha256=fingerprint(args.lock), weights=checkpoint, selection_rule=lock["k10_rule"],
                   eval_sha256=training["inputs"]["eval-dev.jsonl"], corpus_sha256=training["inputs"]["corpus-dev.jsonl"]))
    print(args.out / "lock.json")
