"""Select test operating points using only MS MARCO development measurements."""
import platform
from functools import partial
from time import perf_counter_ns

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .benchmark import EF_SEARCH, HNSW_CONSTRUCTION, create_index, evaluate, project
from .embedding import normalize
from .io import fingerprint, read_json, run_directory, write_json, write_rows
from .projection import development_data, embedding_cache, load_projection

REPRESENTATIONS = ("raw768", "pca128", "learned128")
QUALITY_MARGIN = 0.01
LATENCY_MARGIN = 0.05


def transform(vectors, mean, components):
    return project(normalize(vectors), mean, components)


def measure_p95(index, preprocess, queries, ids, k, order):
    warm = preprocess(queries[order[:16]])
    for row in warm:
        index.search(row[None, :], k)
    milliseconds = []
    for i in order:
        source = queries[i:i+1]
        start = perf_counter_ns()
        _, positions = index.search(preprocess(source), k)
        [ids[p] for p in positions[0]]
        milliseconds.append((perf_counter_ns()-start) / 1e6)
    return float(np.percentile(milliseconds, 95))


def quality_metrics(k):
    return ("ndcg@10", "recall@10") if k == 10 else ("recall@100",)


def choose(points, k, representations=REPRESENTATIONS):
    metrics = quality_metrics(k)
    reference = next(p for p in points if p["representation"] == "raw768" and p["m"] is None and p["k"] == k)
    selected = {}
    for rep in representations:
        candidates = [p for p in points if p["representation"] == rep and p["m"] and p["k"] == k]
        for p in [p for p in points if p["representation"] == rep and p["k"] == k]:
            p["passes_quality"] = all(p["quality"][m] >= reference["quality"][m] - QUALITY_MARGIN - 1e-12
                                      for m in metrics)
        passing = [p for p in candidates if p["passes_quality"]]
        if passing:
            pick = min(passing, key=lambda p: (p["p95_ms"], p["ef_search"]))
        else:
            pick = min(candidates, key=lambda p: (-p["quality"][metrics[0]], p["p95_ms"], p["ef_search"]))
        selected[rep] = pick
    return selected


def select_points(points):
    top10 = choose(points, 10)
    top100 = choose(points, 100)
    reference = top10["raw768"]["p95_ms"]
    matched = {}
    for rep in REPRESENTATIONS:
        candidates = [p for p in points if p["representation"] == rep and p["m"] and p["k"] == 10
                      and abs(p["p95_ms"] / reference - 1) <= LATENCY_MARGIN]
        matched[rep] = min(candidates, default=None, key=lambda p: (
            -p["quality"]["recall@10"], p["p95_ms"], p["ef_search"]))
    return {"k10": top10, "k100": top100, "rq3_latency_matched": matched}


def calibrate(args):
    training = read_json(args.projection / "summary.json")
    cache = embedding_cache(args)
    if cache.settings != training["embedding"]:
        raise ValueError("Encoder settings differ from the trained checkpoint")
    if fingerprint(args.prepared / "manifest.json") != training["prepared_manifest_sha256"]:
        raise ValueError("Development split must match training")
    for name in ("corpus-dev.jsonl", "eval-dev.jsonl"):
        if fingerprint(args.prepared / name) != training["inputs"][name]:
            raise ValueError(f"Changed development input: {name}")
    for name in ("pca.npz", "learned128.npz"):
        if fingerprint(args.projection / name) != training["weights_sha256"][name]:
            raise ValueError(f"Changed projection: {name}")
    if args.repetitions < 5:
        raise ValueError("Use at least five measurements per operating point")

    faiss.omp_set_num_threads(1)
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        ids, documents, dev, queries = development_data(args.prepared, args.passages, cache)
        if len(ids) < 100:
            raise ValueError("Recall@100 requires at least 100 development passages")
        write_rows(out / "passage-ids.jsonl", ids)
        indexes, transforms, points = {}, {}, []
        for rep in REPRESENTATIONS:
            if rep == "raw768":
                preprocess = normalize
            else:
                name = "pca.npz" if rep == "pca128" else "learned128.npz"
                mean, components = load_projection(args.projection / name)
                preprocess = partial(transform, mean=mean, components=components[:128])
            transforms[rep] = preprocess
            indexed = np.empty((len(ids), 768 if rep == "raw768" else 128), dtype=np.float32)
            for start in range(0, len(ids), 8192):
                indexed[start:start+8192] = preprocess(documents[start:start+8192])
            query_vectors = preprocess(queries)
            for m in (None, 32):
                print(f"Building dev {len(ids):,} {rep} m={m}", flush=True)
                index, index_name, costs = create_index(indexed, rep, out, args.seed, m)
                indexes[index_name] = index
                for k in (10, 100):
                    search_depths = [n for n in EF_SEARCH if n >= k] if m else [None]
                    for ef in search_depths:
                        if m:
                            index.hnsw.efSearch = ef
                        records, _, quality = evaluate(index, ids, [("dev", dev)], query_vectors, k, None)
                        name = f"{index_name}-k{k}" + (f"-ef{ef}" if m else "")
                        write_rows(out / f"{name}.jsonl", records)
                        points.append({"name": name, "representation": rep, "index": index_name,
                                       "ef_search": ef, "k": k, "quality": quality["dev"],
                                       "repeated_p95_ms": [], **costs})
            del indexed
        del documents
        rng = np.random.default_rng(args.seed)
        rounds = []
        for repeat in range(args.repetitions):
            order = rng.permutation(len(queries))
            configurations = rng.permutation(len(points))
            rounds.append([points[i]["name"] for i in configurations])
            for i in configurations:
                p = points[i]
                index = indexes[p["index"]]
                if p["m"]:
                    index.hnsw.efSearch = p["ef_search"]
                p["repeated_p95_ms"].append(measure_p95(index, transforms[p["representation"]], queries,
                                                        ids, p["k"], order))
            print(f"Timing round {repeat+1}/{args.repetitions} complete", flush=True)
        for p in points:
            p["p95_ms"] = float(np.median(p["repeated_p95_ms"]))
        selected = select_points(points)
        write_json(out / "summary.json", {
            "passages": len(ids), "queries": len(queries), "split": "dev",
            "query_file_sha256": training["inputs"]["eval-dev.jsonl"],
            "projection_summary_sha256": fingerprint(args.projection / "summary.json"),
            "configurations": points, "selected": selected, "measurement_order": rounds,
            "quality_margin": QUALITY_MARGIN, "latency_matching_fraction": LATENCY_MARGIN,
            "timing": "median of round-wise single-query pipeline p95 milliseconds",
            "seed": args.seed, "threads": 1, "ef_construction": HNSW_CONSTRUCTION,
            "faiss": faiss.__version__, "numpy": np.__version__, "platform": platform.platform(),
        })
    print(args.out / "summary.json", flush=True)


def locked_setting(point):
    if point is None:
        return None
    return {k: point[k] for k in ("name", "m", "ef_search", "k", "quality", "p95_ms", "passes_quality")}


def lock(args):
    training = read_json(args.projection / "summary.json")
    training_digest = fingerprint(args.projection / "summary.json")
    if args.out.exists():
        raise FileExistsError(f"Choose a new lock path: {args.out}")
    for name, digest in training["weights_sha256"].items():
        if fingerprint(args.projection / name) != digest:
            raise ValueError(f"Changed checkpoint: {name}")
    sizes = {}
    calibration_files = {}
    seeds = set()
    for path in args.calibration:
        file = path / "summary.json"
        result = read_json(file)
        if (result["split"] != "dev" or result["projection_summary_sha256"] != training_digest
                or result["query_file_sha256"] != training["inputs"]["eval-dev.jsonl"]):
            raise ValueError("A calibration does not match this training run")
        size = result["passages"]
        if size in sizes:
            raise ValueError(f"Duplicate calibration for {size}")
        seeds.add(result["seed"])
        selected = select_points(result["configurations"])
        if selected != result["selected"]:
            raise ValueError("Saved operating points differ from selection rules")
        settings = {name: {rep: locked_setting(p) for rep, p in values.items()}
                    for name, values in selected.items()}
        settings["flat_k10"] = {p["representation"]: locked_setting(p) for p in result["configurations"]
                                if p["m"] is None and p["k"] == 10}
        sizes[size] = settings
        calibration_files[size] = {"file": str(file), "sha256": fingerprint(file)}
    if len(seeds) != 1:
        raise ValueError("Use a common index seed at every scale")
    write_json(args.out, {
        "selection_split": "MS MARCO dev", "sizes": sizes, "rq1_primary_passages": max(sizes),
        "corpus_membership": "First N rows of the corresponding prepared corpus-dev.jsonl or corpus-trec.jsonl",
        "projection": {"file": str(args.projection / "learned128.npz"),
                       "sha256": training["weights_sha256"]["learned128.npz"],
                       "dimension": 128, "parameters": 98304,
                       "transform": "L2Norm(W @ (L2Norm(x) - mean))", "hyperparameters": training["selected"]},
        "pca": {"file": str(args.projection / "pca.npz"), "components": "first 128 rows",
                "sha256": training["weights_sha256"]["pca.npz"]},
        "training_summary": {"file": str(args.projection / "summary.json"), "sha256": training_digest},
        "calibration": calibration_files, "ef_construction": HNSW_CONSTRUCTION,
        "build_threads": 1, "search_threads": 1, "index_seed": seeds.pop(),
        "quality_margin": QUALITY_MARGIN, "latency_matching_fraction": LATENCY_MARGIN,
        "k10_rule": "min p95 within both nDCG@10 and Recall@10 margins of same-size raw flat; fallback max nDCG then min p95",
        "k100_rule": "min p95 within Recall@100 margin of same-size raw flat; fallback max Recall@100 then min p95",
        "formal_test_grid": {
            "representations": list(REPRESENTATIONS), "primary_m": 32,
            "ef_search": {str(k): [ef for ef in EF_SEARCH if ef >= k] for k in (10, 100)},
            "m_sensitivity_at_100000": [16, 32, 48],
        },
        "full_corpus_followup": "raw768, pca128, learned128; FlatIP and HNSW M32; same-size dev calibration required before test",
        "hotpot": "Use k10 settings at the matching passage count",
    })
    print(args.out)
