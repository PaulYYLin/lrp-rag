"""Build and benchmark the locked TREC test grid."""
import platform
from functools import partial
from itertools import islice
from time import perf_counter_ns
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .benchmark import create_index, evaluate, HNSW_CONSTRUCTION
from .calibration import transform
from .embedding import normalize
from .io import fingerprint, read_json, read_rows, run_directory, write_json, write_rows
from .projection import embedding_cache, load_projection


def verified_file(record):
    path = Path(record["file"])
    if fingerprint(path) != record["sha256"]:
        raise ValueError(f"Locked input changed: {path}")
    return path


def locked_encoder(args):
    lock = read_json(args.lock)
    if str(args.passages) not in lock["sizes"]:
        raise ValueError("No development operating points for this size")
    if HNSW_CONSTRUCTION != lock["ef_construction"]:
        raise ValueError("Graph construction differs from the lock")
    training = read_json(verified_file(lock["training_summary"]))
    cache = embedding_cache(args)
    if cache.settings != training["embedding"]:
        raise ValueError("Use the encoder settings recorded during training")
    transforms = {"raw768": normalize}
    for rep, key in (("pca128", "pca"), ("learned128", "projection")):
        mean, components = load_projection(verified_file(lock[key]))
        transforms[rep] = partial(transform, mean=mean, components=components[:128])
    return lock, cache, transforms, training


def configuration(args):
    lock, cache, transforms, training = locked_encoder(args)
    if fingerprint(args.prepared / "manifest.json") != training["prepared_manifest_sha256"]:
        raise ValueError("Prepared split differs from training")
    return lock, cache, transforms


def groups_for(prepared):
    return [(year, list(read_rows(prepared / f"eval-{year}.jsonl")))
            for year in ("trec2019", "trec2020")]


def build(args):
    lock, cache, transforms = configuration(args)
    groups = groups_for(args.prepared)
    corpus_file = args.prepared / "corpus-trec.jsonl"
    corpus = list(islice(read_rows(corpus_file), args.passages))
    ids = [row["id"] for row in corpus]
    judged = {pid for _, rows in groups for row in rows for pid in row["qrels"]}
    if len(ids) != args.passages or len(set(ids)) != len(ids) or judged.difference(ids):
        raise ValueError("Invalid TREC corpus membership")
    grid = lock["formal_test_grid"]
    links = grid["m_sensitivity_at_100000"] if args.passages == 100000 else [grid["primary_m"]]
    faiss.omp_set_num_threads(lock["build_threads"])
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        write_rows(out / "passage-ids.jsonl", ids)
        docs = cache.vectors(corpus, "TREC corpus")
        del corpus
        queries = np.concatenate([cache.vectors(rows, year) for year, rows in groups])
        costs, points, full = {}, [], {}
        for rep in grid["representations"]:
            preprocess = transforms[rep]
            vectors = np.empty((len(ids), 768 if rep == "raw768" else 128), dtype="float32")
            for start in range(0, len(ids), 8192):
                vectors[start:start+8192] = preprocess(docs[start:start+8192])
            q = preprocess(queries)
            exact = {}
            for m in [None, *links]:
                print(f"Building {args.passages:,} {rep} M={m}", flush=True)
                index, name, cost = create_index(vectors, rep, out, lock["index_seed"], m)
                costs[name] = cost
                for k in (10, 100):
                    for ef in grid["ef_search"][str(k)] if m else [None]:
                        if m:
                            index.hnsw.efSearch = ef
                        records, rankings, quality = evaluate(
                            index, ids, groups, q, k, exact.get(k) if m else None,
                            full.get(k) if not m and rep != "raw768" else None)
                        if not m:
                            exact[k] = rankings
                            if rep == "raw768":
                                full[k] = rankings
                        operation = f"{name}-k{k}" + (f"-ef{ef}" if m else "")
                        write_rows(out / f"{operation}.jsonl", records)
                        points.append(dict(name=operation, index=name, representation=rep, k=k,
                                           ef_search=ef, m=m, quality=quality))
                del index
        write_json(out / "build.json", dict(
            passages=len(ids), judged=len(judged), background=len(ids)-len(judged),
            lock_sha256=fingerprint(args.lock), corpus_sha256=fingerprint(corpus_file),
            evaluation_sha256={y: fingerprint(args.prepared / f"eval-{y}.jsonl") for y, _ in groups},
            indexes=costs, configurations=points, faiss=faiss.__version__, platform=platform.platform()))


def resident_index(path):
    """Measure current RSS before and after deserializing one index in a fresh interpreter."""
    import subprocess
    import sys

    program = """
import os, subprocess, sys
import faiss

def rss():
    # ps reports KiB on macOS and Linux.
    return 1024 * int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())]))

before = rss()
index = faiss.read_index(sys.argv[1])
print(before, rss())
"""
    before, after = map(int, subprocess.check_output([sys.executable, "-c", program, str(path)]).split())
    return dict(process_before_bytes=before, process_loaded_bytes=after, rss_delta_bytes=after-before)


def retrieve(index, preprocess, vectors, ids, k):
    _, positions = index.search(preprocess(vectors), k)
    return [[ids[p] for p in row] for row in positions]


def quality_for_batches(out, point, rows, rankings, batch_size):
    from .benchmark import metrics

    k = point["k"]
    entries = []
    for (split, row), ranking in zip(rows, rankings, strict=True):
        if len(set(ranking)) != k:
            raise ValueError("Invalid measured ranking")
        entries.append(dict(split=split, id=row["id"], passage_ids=ranking,
                            metrics=metrics(ranking, row["qrels"], k, 2)))
    write_rows(out / f"{point['name']}-batch{batch_size}.jsonl", entries)
    result = {}
    for split in ("trec2019", "trec2020", "trec_combined"):
        group = [r for r in entries if split == "trec_combined" or r["split"] == split]
        result[split] = {key: float(np.mean([r["metrics"][key] for r in group]))
                         for key in entries[0]["metrics"]}
    return result


def time_point(index, preprocess, queries, ids, k, order, capture):
    projected = preprocess(queries)
    for i in order[:16]:
        index.search(projected[i:i+1], k)
    search, single, batch = [], [], []
    rankings = {b: [None] * len(queries) for b in (1, 32)} if capture else {}
    for i in order:
        source = queries[i:i+1]
        start = perf_counter_ns()
        found = retrieve(index, preprocess, source, ids, k)
        single.append((perf_counter_ns() - start) / 1e6)
        if capture:
            rankings[1][i] = found[0]
    for i in order:
        vector = projected[i:i+1]
        start = perf_counter_ns()
        index.search(vector, k)
        search.append((perf_counter_ns() - start) / 1e6)
    # Cycle from the beginning to fill the last batch. Count every timed request.
    padded = np.resize(order, 32 * ((len(order) + 31) // 32)).reshape(-1, 32)
    retrieve(index, preprocess, queries[padded[0]], ids, k)
    for positions in padded:
        source = queries[positions]
        start = perf_counter_ns()
        found = retrieve(index, preprocess, source, ids, k)
        batch.append((perf_counter_ns() - start) / 1e6)
        if capture:
            for i, ranking in zip(positions, found, strict=True):
                if rankings[32][i] is None:
                    rankings[32][i] = ranking
    result = {}
    for name, values in (("search", search), ("pipeline", single), ("batch32", batch)):
        result[f"{name}_p50_ms"], result[f"{name}_p95_ms"] = map(float, np.percentile(values, (50, 95)))
    result["batch32_queries_per_second"] = padded.size * 1000 / sum(batch)
    result["single_queries_per_second"] = len(order) * 1000 / sum(single)
    return result, rankings


def timing_host():
    import os
    import subprocess

    host = {"platform": platform.platform(), "cpu": platform.machine()}
    if platform.system() == "Darwin":
        from ctypes import CDLL

        host["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        host["physical_memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        # threadpoolctl does not enumerate Apple's Accelerate BLAS.
        accelerate = CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        if hasattr(accelerate, "BLASSetThreading"):
            if accelerate.BLASSetThreading(1) or accelerate.BLASGetThreading() != 1:
                raise RuntimeError("Could not select single-threaded Accelerate BLAS")
            host["accelerate_blas"] = "single-threaded on the calling thread"
    elif "SC_PHYS_PAGES" in os.sysconf_names:
        host["physical_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return host


def measure(args):
    lock, cache, transforms = configuration(args)
    build_info = read_json(args.out / "build.json")
    groups = groups_for(args.prepared)
    if build_info["lock_sha256"] != fingerprint(args.lock) or build_info["passages"] != args.passages:
        raise ValueError("Index build and measurement lock or passage count differ")
    for year, _ in groups:
        if fingerprint(args.prepared / f"eval-{year}.jsonl") != build_info["evaluation_sha256"][year]:
            raise ValueError("Test inputs have changed since index construction")
    rows = [(year, row) for year, group in groups for row in group]
    ids = list(read_rows(args.out / "passage-ids.jsonl"))
    queries = np.concatenate([cache.vectors(group, year) for year, group in groups])
    settings = lock["sizes"][str(args.passages)]
    selected = {p["name"] for group in ("k10", "flat_k10") for p in settings[group].values()}
    faiss.omp_set_num_threads(lock["search_threads"])
    host = timing_host()
    rng = np.random.default_rng(lock["index_seed"])
    points = build_info["configurations"]
    with run_directory(args.out / "measurement") as out, threadpool_limits(limits=1, user_api="blas"):
        indexes = {}
        for name, cost in build_info["indexes"].items():
            path = args.out / f"{name}.faiss"
            cost.update(resident_index(path))
            indexes[name] = faiss.read_index(str(path))
        orders = []
        for round_id in range(5):
            order = rng.permutation(len(rows))
            sequence = rng.permutation(len(points))
            orders.append(dict(points=[points[i]["name"] for i in sequence],
                               queries=[rows[i][1]["id"] for i in order]))
            for i in sequence:
                point = points[i]
                index = indexes[point["index"]]
                if point["m"]:
                    index.hnsw.efSearch = point["ef_search"]
                times, rankings = time_point(index, transforms[point["representation"]], queries,
                                             ids, point["k"], order, round_id == 0)
                point.setdefault("rounds", []).append(times)
                if rankings:
                    point["batch_quality"] = {
                        str(b): quality_for_batches(out, point, rows, result, b)
                        for b, result in rankings.items()}
            print(f"{args.passages:,} retrieval timing round {round_id+1}/5", flush=True)

        # Measure the actual uncached query path independently of retrieval-only timing.
        cache.encode([rows[0][1]["text"]])
        by_name = {p["name"]: p for p in points}
        by_id = {row["id"]: row["text"] for _, row in rows}
        for r, orders_r in enumerate(orders):
            for name in orders_r["points"]:
                if name not in selected:
                    continue
                point = by_name[name]
                index = indexes[point["index"]]
                if point["m"]:
                    index.hnsw.efSearch = point["ef_search"]
                preprocess = transforms[point["representation"]]
                durations = []
                for qid in orders_r["queries"]:
                    text = by_id[qid]
                    start = perf_counter_ns()
                    retrieve(index, preprocess, cache.encode([text]), ids, 10)
                    durations.append((perf_counter_ns()-start)/1e6)
                point["rounds"][r].update(zip(("encoding_pipeline_p50_ms", "encoding_pipeline_p95_ms"),
                                               map(float, np.percentile(durations, [50, 95])), strict=True))
            print(f"{args.passages:,} uncached query timing round {r+1}/5", flush=True)
        for point in points:
            point["median"] = {key: float(np.median([x[key] for x in point["rounds"]]))
                               for key in point["rounds"][0]}
        write_json(out / "summary.json", {**build_info, "order": orders, "host": host,
                   "search_threads": lock["search_threads"], "numpy": np.__version__,
                   "build_sha256": fingerprint(args.out / "build.json"), "embedding": cache.settings})
