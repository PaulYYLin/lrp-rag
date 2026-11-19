import math
import platform
from collections import defaultdict
from itertools import batched
from time import perf_counter, perf_counter_ns

import faiss
import numpy as np

from .embedding import normalize
from .io import atomic_file, write_json, write_rows

EF_SEARCH = (10, 16, 32, 64, 128, 256, 512)
CUTOFFS = (10, 100)
HNSW_CONSTRUCTION = 200


def metrics(ranking, judgments, k, relevant_grade):
    relevant = {pid for pid, grade in judgments.items() if grade >= relevant_grade}
    recall = len(relevant.intersection(ranking)) / len(relevant) if relevant else 0.0
    values = {f"recall@{k}": recall}
    if k == 10:
        def dcg(grades):
            return sum(grade / math.log2(rank + 2) for rank, grade in enumerate(grades))

        ideal = dcg(sorted(judgments.values(), reverse=True)[:k])
        actual = dcg(judgments.get(pid, 0) for pid in ranking)
        values["ndcg@10"] = actual / ideal if ideal else 0.0
    return values


def project(vectors, mean, components):
    return normalize((vectors - mean) @ components.T)


def fit_pca(query_vectors, document_vectors, dimensions, out, query_ids, document_ids):
    if len(query_vectors) != len(document_vectors) or len(query_vectors) * 2 <= dimensions:
        raise ValueError("PCA needs balanced training samples and more rows than requested dimensions")
    training = np.concatenate((normalize(query_vectors), normalize(document_vectors))).astype(np.float64)
    mean = training.mean(axis=0)
    training -= mean
    eigenvalues, directions = np.linalg.eigh(training.T @ training / (len(training) - 1))
    if eigenvalues[-dimensions] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Training embeddings do not span the requested PCA dimensions")
    components = directions[:, -dimensions:][:, ::-1].T
    anchors = np.argmax(np.abs(components), axis=1)
    components *= np.sign(components[np.arange(dimensions), anchors])[:, None]
    mean = mean.astype(np.float32)
    components = np.ascontiguousarray(components, dtype=np.float32)
    with atomic_file(out / "pca.npz", "wb") as stream:
        np.savez(stream, mean=mean, components=components,
                 explained_variance_ratio=eigenvalues[-dimensions:][::-1] / eigenvalues.sum(),
                 query_ids=query_ids, passage_ids=document_ids)
    return mean, components


def create_index(documents, representation, out, seed, m):
    dimensions = documents.shape[1]
    start = perf_counter()
    if m is None:
        index = faiss.IndexFlatIP(dimensions)
        name = f"{representation}-flat"
    else:
        index = faiss.IndexHNSWFlat(dimensions, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_CONSTRUCTION
        index.hnsw.rng = faiss.RandomGenerator(seed)
        name = f"{representation}-hnsw-m{m}"
    for start_row in range(0, len(documents), 8192):
        index.add(documents[start_row:start_row + 8192])
    cost = perf_counter() - start
    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))
    return index, name, {"build_seconds": cost, "index_bytes": path.stat().st_size, "m": m}


def timings(index, embeddings, preprocess, ids, k, repetitions, seed):
    projected = preprocess(embeddings)
    order = np.arange(len(embeddings))
    index.search(projected[:min(16, len(projected))], k)
    for row in projected[:min(16, len(projected))]:
        index.search(row[None, :], k)

    repeats = []
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        rng.shuffle(order)
        search_times = []
        pipeline_times = []
        for i in order:
            query = projected[i:i + 1]
            source = embeddings[i:i + 1]
            start = perf_counter_ns()
            index.search(query, k)
            search_times.append((perf_counter_ns() - start) / 1e6)
            start = perf_counter_ns()
            _, neighbors = index.search(preprocess(source), k)
            [ids[position] for position in neighbors[0] if position >= 0]
            pipeline_times.append((perf_counter_ns() - start) / 1e6)

        start = perf_counter()
        for batch in batched(order, 32):
            _, neighbors = index.search(preprocess(embeddings[list(batch)]), k)
            [[ids[position] for position in row if position >= 0] for row in neighbors]
        throughput = len(embeddings) / (perf_counter() - start)
        repeats.append({
            "pipeline_p50_ms": float(np.percentile(pipeline_times, 50)),
            "pipeline_p95_ms": float(np.percentile(pipeline_times, 95)),
            "search_p50_ms": float(np.percentile(search_times, 50)),
            "search_p95_ms": float(np.percentile(search_times, 95)),
            "batch32_queries_per_second": throughput,
        })
    return {"repetitions": repeats,
            "median": {key: float(np.median([row[key] for row in repeats])) for key in repeats[0]}}


def evaluate(index, ids, groups, query_vectors, k, reference, projection_reference=None):
    result = []
    rankings = []
    offset = 0
    for split, rows in groups:
        group_vectors = query_vectors[offset:offset + len(rows)]
        scores, neighbors = index.search(group_vectors, k)
        for row, distances, positions in zip(rows, scores, neighbors, strict=True):
            if len(set(positions)) != k or np.any(positions < 0):
                raise ValueError(f"Invalid top-{k} results for {split}/{row['id']}")
            ranking = [ids[position] for position in positions]
            entry = {"split": split, "id": row["id"], "passage_ids": ranking,
                     "scores": distances.tolist(), "metrics": {}}
            if "qrels" in row:
                grade = 2 if split.startswith("trec") else 1
                entry["metrics"] = metrics(ranking, row["qrels"], k, grade)
            else:
                mapped = {fact["page_id"] for fact in row["supporting_facts"] if fact["status"] == "mapped"}
                annotated = {fact["gold_title"] for fact in row["supporting_facts"]}
                covered = len(mapped.intersection(ranking))
                entry["metrics"][f"mapped_support_pages_retrieved@{k}"] = covered
                entry["metrics"][f"support_page_recall@{k}"] = min(covered, len(annotated)) / len(annotated)
            i = len(rankings)
            for name, baseline in (("ann_recall", reference), ("projection_overlap", projection_reference)):
                if baseline is not None:
                    entry["metrics"][f"{name}@{k}"] = len(set(ranking).intersection(baseline[i])) / k
            rankings.append(ranking)
            result.append(entry)
        offset += len(rows)

    summaries = defaultdict(list)
    for record in result:
        summaries[record["split"]].append(record["metrics"])
    if len(groups) > 1:
        summaries["trec_combined"] = [record["metrics"] for record in result]
    return result, rankings, {
        split: {"queries": len(values), **{
            key: sum(row[key] for row in values) / len(values) for key in values[0]
        }} for split, values in summaries.items()
    }


def run(out, rows, documents, groups, query_vectors, pca, seed, threads, repetitions, links,
        cutoffs=CUTOFFS, ef_values=EF_SEARCH):
    ids = [row["id"] for row in rows]
    if len(ids) < max(cutoffs) or len(ids) != len(set(ids)):
        raise ValueError("Corpus must have unique passage IDs and at least max(k) rows")
    if not groups or not all(group for _, group in groups):
        raise ValueError("Each evaluation split must have at least one query")

    write_rows(out / "passage-ids.jsonl", ids)
    configurations = []
    full_rankings = {}
    for representation, mean_components in (("raw768", None), (f"pca{pca[1].shape[0]}", pca)):
        if mean_components is None:
            preprocess = normalize
            indexed = np.empty_like(documents)
            for start in range(0, len(documents), 8192):
                indexed[start:start + 8192] = normalize(documents[start:start + 8192])
        else:
            mean, components = mean_components

            def preprocess(vectors):
                return project(normalize(vectors), mean, components)

            indexed = np.empty((len(documents), len(components)), dtype=np.float32)
            for start in range(0, len(documents), 8192):
                indexed[start:start + 8192] = preprocess(documents[start:start + 8192])
        transformed = preprocess(query_vectors)
        exact = {}
        for m in [None, *links]:
            # A single add thread fixes HNSW's insertion order and scheduling.
            faiss.omp_set_num_threads(1)
            print(f"Building {representation} {'FlatIP' if m is None else f'HNSW M={m}'}", flush=True)
            index, name, costs = create_index(indexed, representation, out, seed, m)
            faiss.omp_set_num_threads(threads)
            for k in cutoffs:
                for ef in [None] if m is None else [point for point in ef_values if point >= k]:
                    if ef is not None:
                        index.hnsw.efSearch = ef
                    reference = exact.get(k) if m is not None else None
                    projection_reference = full_rankings.get(k) if m is None and mean_components is not None else None
                    result, rankings, quality = evaluate(
                        index, ids, groups, transformed, k, reference, projection_reference
                    )
                    if m is None:
                        exact[k] = rankings
                    if m is None and mean_components is None:
                        full_rankings[k] = rankings
                    operation = f"{name}-k{k}" + (f"-ef{ef}" if ef is not None else "")
                    write_rows(out / f"{operation}.jsonl", result)
                    configurations.append({"name": operation, "dimensions": indexed.shape[1],
                                           "index": name, "k": k, "ef_search": ef,
                                           "quality": quality, **costs,
                                           "timing": timings(index, query_vectors, preprocess, ids, k,
                                                             repetitions, seed)})
            del index
    write_json(out / "summary.json", {
        "passages": len(rows), "seed": seed, "search_threads": threads, "build_threads": 1,
        "ef_construction": HNSW_CONSTRUCTION, "faiss": faiss.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "ndcg_gain": "linear",
        "configurations": configurations,
    })
