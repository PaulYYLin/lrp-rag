"""Compare HotpotQA quality with warm-index and live-encoder retrieval latency."""
from time import perf_counter_ns

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .io import fingerprint, read_json, read_rows, run_directory, sample_key, write_json
from .phase3 import locked_encoder, retrieve, time_point, timing_host
from .phase3_analysis import latency_ratio
from .phase4_256 import learned256, pca256


def plot(out, points, quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = dict(raw768="#28649a", pca128="#777777", pca256="#85468e", learned128="#c06a1a", learned256="#168376")
    labels = dict(raw768="Original 768", pca128="PCA 128", pca256="PCA 256", learned128="Learned 128", learned256="Learned 256")
    representations = dict.fromkeys(name.split("-")[0] for name in points)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, metric, label in zip(axes, ("complete", "f1"), ("Complete evidence (%)", "Answer token F1 (%)")):
        for name, point in points.items():
            rep, mode = name.split("-", maxsplit=1)
            ax.scatter(point["median"]["pipeline_p95_ms"], 100*quality[name]["all"][metric],
                       s=77, color=colors[rep], marker="o" if mode == "flat" else "^",
                       edgecolors="white", linewidths=0.6)
        ax.set(xscale="log", xlabel="Retrieval p95 (ms, log scale)", ylabel=label)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        ax.grid(alpha=.23)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[rep], lw=3, label=labels[rep]) for rep in representations]
    handles.extend(Line2D([], [], color="#333333", marker=m, linestyle="", label=s)
                   for m, s in (("o", "Exact"), ("^", "HNSW")))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.25, wspace=.29)
    fig.savefig(out / "quality-latency.png", dpi=180)
    plt.close(fig)


def measure(args):
    dirs = {"retrieval": args.retrieval, "retrieval256": args.retrieval256}
    quality = read_json(args.report / "summary.json")
    data = read_json(args.prepared / "manifest.json")
    args.passages = data["corpus_passages"]
    lock, cache, transforms, training = locked_encoder(args)
    transforms["learned256"], _ = learned256(lock, training)
    if args.retrieval_pca256:
        dirs["retrieval_pca256"] = args.retrieval_pca256
        transforms["pca256"] = pca256(lock)
    qs = list(read_rows(args.prepared / "eval-hotpot.jsonl"))
    qids = [q["id"] for q in qs]
    vectors = cache.vectors(qs, "Hotpot timing queries")
    ids = list(read_rows(args.retrieval / "passage-ids.jsonl"))
    indexes, points, reference = {}, {}, {}
    for label, directory in dirs.items():
        meta = read_json(directory / "manifest.json")
        if (fingerprint(directory / "manifest.json") != quality["input_manifests"][label]
                or meta["lock_sha256"] != fingerprint(args.lock)
                or meta["questions_sha256"] != fingerprint(args.prepared / "eval-hotpot.jsonl")
                or ids != list(read_rows(directory / "passage-ids.jsonl"))):
            raise ValueError("Quality results and timing inputs differ")
        for name, point in meta["points"].items():
            index = faiss.read_index(str(directory / f"{name}.faiss"))
            if point["m"]:
                index.hnsw.efSearch = point["ef_search"]
            indexes[name] = index
            points[name] = dict(rounds=[], index_bytes=point["index_bytes"], ef_search=point["ef_search"])
            path = directory / f"{name}.jsonl"
            if fingerprint(path) != meta["rankings_sha256"][name]:
                raise ValueError("Quality rankings changed")
            rows = list(read_rows(path))
            if [r["id"] for r in rows] != qids:
                raise ValueError("Query order changed")
            reference[name] = [r["passage_ids"] for r in rows]
    if set(indexes) != set(quality["conditions"]) - {"question_only", "gold"}:
        raise ValueError("Timing must cover all retrieval conditions in the quality report")
    seed = lock["index_seed"]
    rng = np.random.default_rng(seed)
    # One common, preselected 100-question sample for direct text-to-ID timing.
    live = sorted(range(len(qs)), key=lambda i: sample_key(seed, "hotpot-live-latency", qids[i]))[:100]
    host = timing_host()
    faiss.omp_set_num_threads(lock["search_threads"])
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        orders = []
        for round_id in range(5):
            order = rng.permutation(len(qs))
            sequence = rng.permutation(sorted(indexes)).tolist()
            orders.append(dict(methods=sequence, questions=[qids[i] for i in order]))
            for name in sequence:
                prep = transforms[name.split("-")[0]]
                times, rankings = time_point(indexes[name], prep, vectors, ids, 10, order, round_id == 0)
                points[name]["rounds"].append(times)
                if rankings:
                    points[name]["ranking_differences"] = {
                        str(b): [dict(id=qids[i], passage_ids=r) for i, r in enumerate(ranks)
                                 if r != reference[name][i]] for b, ranks in rankings.items()}
            print(f"Hotpot warm-index timing round {round_id+1}/5", flush=True)
        cache.encode([qs[live[0]]["text"]])
        for round_id in range(5):
            live_order = rng.permutation(live)
            sequence = rng.permutation(sorted(indexes)).tolist()
            orders[round_id].update(live_methods=sequence, live_questions=[qids[i] for i in live_order])
            for name in sequence:
                preprocess = transforms[name.split("-")[0]]
                elapsed = []
                for i in live_order:
                    start = perf_counter_ns()
                    fresh = cache.encode([qs[i]["text"]])
                    retrieve(indexes[name], preprocess, fresh, ids, 10)
                    elapsed.append((perf_counter_ns()-start)/1e6)
                points[name]["rounds"][round_id]["text_to_ids_p95_ms"] = float(np.percentile(elapsed, 95))
            print(f"Hotpot text-to-ID timing round {round_id+1}/5", flush=True)
        for point in points.values():
            point["median"] = {k: float(np.median([r[k] for r in point["rounds"]])) for k in point["rounds"][0]}
        comparisons = {}
        for name, p in points.items():
            comparisons[name] = {}
            for ref in ("raw768-flat", "raw768-hnsw-m32", "learned128-flat", "learned256-flat"):
                b = points[ref]
                ratios = {metric: latency_ratio(p, b, metric) for metric in ("pipeline_p95_ms", "text_to_ids_p95_ms")}
                comparisons[name][ref] = dict(candidate_over_reference=ratios,
                    p95_ms_saved=b["median"]["pipeline_p95_ms"]-p["median"]["pipeline_p95_ms"])
        write_json(out / "timing.json", dict(points=points, comparisons=comparisons, host=host, seed=seed,
                   search_threads=lock["search_threads"], faiss=faiss.__version__, orders=orders,
                   queries=len(qs), live_queries=len(live), batch_size=32, rounds=5,
                   quality_sha256=fingerprint(args.report / "summary.json")))
        plot(out, points, quality["conditions"])
