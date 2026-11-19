"""Fit shared, centered linear maps and compare them on held-out MS MARCO queries."""
from itertools import islice, product
from time import perf_counter

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from .benchmark import evaluate, fit_pca, project
from .embedding import EmbeddingCache, normalize
from .io import atomic_file, fingerprint, read_json, read_rows, run_directory, write_json, write_rows


def development_data(prepared, size, cache):
    sizes = read_json(prepared / "manifest.json")["sizes"]
    if not min(sizes) <= size <= max(sizes):
        raise ValueError("Choose a development corpus prefix within the prepared size range")
    corpus = list(islice(read_rows(prepared / "corpus-dev.jsonl"), size))
    dev = list(read_rows(prepared / "eval-dev.jsonl"))
    ids = [row["id"] for row in corpus]
    judged = {pid for row in dev for pid in row["qrels"]}
    if len(set(ids)) != size or not dev or not judged.issubset(ids):
        raise ValueError("Development corpus incomplete or missing judged passages")
    return ids, cache.vectors(corpus, "dev passages"), dev, cache.vectors(dev, "dev queries")


def embedding_cache(args):
    return EmbeddingCache(args.cache, args.device, 64, 128, 1, args.model_dir)


def save_projection(path, mean, components):
    with atomic_file(path, "wb") as stream:
        np.savez(stream, mean=mean, components=components)


def load_projection(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        return checkpoint["mean"], checkpoint["components"]


def rank_exact(context, weights, save=None):
    ids, passages, dev, queries = context
    if weights is not None:
        passages = project(passages, *weights)
        queries = project(queries, *weights)
    index = faiss.IndexFlatIP(passages.shape[1])
    index.add(passages)
    records, _, result = evaluate(index, ids, [("dev", dev)], queries, 10, None)
    if save is not None:
        write_rows(save, records)
    return result["dev"]


def paired_dev_intervals(out, results, seed):
    scores = {name: {row["id"]: row["metrics"] for row in read_rows(out / f"{name}-k10.jsonl")}
              for name in results}
    ids = list(scores["raw768"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(10000, len(ids)), dtype=np.int32)
    contrasts = (("learned128", "pca128"), ("learned128", "raw768"), ("learned128", "random128"),
                 ("learned128", "inbatch128"), ("learned64", "pca64"), ("learned256", "pca256"))
    comparisons = {}
    for a, b in contrasts:
        entry = {}
        for metric in ("ndcg@10", "recall@10"):
            differences = np.array([scores[a][qid][metric] - scores[b][qid][metric] for qid in ids])
            interval = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
            entry[metric] = {"difference": float(differences.mean()), "95_percentile_interval": interval.tolist()}
        comparisons[f"{a} minus {b}"] = entry
    write_json(out / "paired-dev.json", {
        "split": "dev", "queries": len(ids), "seed": seed, "paired_bootstrap_samples": len(draws),
        "interval": "query-level percentile; conditional on selected checkpoint", "comparisons": comparisons,
    })


def batch_loss(weights, query_vectors, doc_vectors, batch, positives, selected, negatives, temperature):
    # Unique in-batch passage IDs provide one softmax column per passage.
    candidate_ids, targets = np.unique(selected[batch], return_inverse=True)
    queries = F.normalize(F.linear(query_vectors[batch], weights))
    docs = F.normalize(F.linear(doc_vectors[candidate_ids], weights))
    scores = queries @ docs.T
    columns = {pid: i for i, pid in enumerate(candidate_ids)}
    mask = np.zeros(scores.shape, dtype=bool)
    for i, q in enumerate(batch):
        for pid in positives[q]:
            if pid != selected[q] and pid in columns:
                mask[i, columns[pid]] = True
    scores = scores.masked_fill(torch.from_numpy(mask), -torch.inf)
    if negatives is not None:
        hard = negatives[batch]
        # -1 pads shorter candidate lists. Known positives were removed before mining.
        hard_docs = F.normalize(F.linear(doc_vectors[hard.clip(min=0)], weights), dim=-1)
        hard_scores = (queries[:, None, :] * hard_docs).sum(-1)
        duplicates = (hard < 0) | np.isin(hard, candidate_ids)
        scores = torch.cat((scores, hard_scores.masked_fill(torch.from_numpy(duplicates), -torch.inf)), dim=1)
    return F.cross_entropy(scores / temperature, torch.from_numpy(targets))


def fit_trial(data, initial, epochs, lr, temperature, hard, seed, batch_size, validate):
    q, d, positives, negatives = data
    weights = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer = torch.optim.Adam([weights], lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        order = rng.permutation(len(q))
        selected = np.array([p[(epoch-1) % len(p)] for p in positives])
        loss_sum = 0.0
        for offset in range(0, len(q), batch_size):
            batch = order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(weights, q, d, batch, positives, selected, negatives if hard else None, temperature)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        entry = {"epoch": epoch, "lr": lr, "temperature": temperature, "loss": loss_sum / len(q),
                 "training_seconds": perf_counter()-start}
        validate(entry, weights.detach().numpy())
        history.append(entry)
    return history


def train(args):
    from .embedding import DIMENSION

    if args.batch_size < 2 or min(args.learning_rates + args.temperatures) <= 0:
        raise ValueError("Use batch size >= 2 and positive learning rates and temperatures")
    train_meta = read_json(args.training / "manifest.json")
    prepared_hash = fingerprint(args.prepared / "manifest.json")
    if prepared_hash != train_meta["prepared_manifest_sha256"]:
        raise ValueError("Training and development data come from different preparations")
    for name, digest in train_meta["files"].items():
        if fingerprint(args.training / name) != digest:
            raise ValueError(f"Training input changed: {name}")

    cache = embedding_cache(args)
    torch.use_deterministic_algorithms(True)
    faiss.omp_set_num_threads(1)
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        fit_queries = list(read_rows(args.prepared / "pca-queries.jsonl"))
        fit_docs = list(read_rows(args.prepared / "pca-passages.jsonl"))
        mean, pca = fit_pca(cache.vectors(fit_queries, "PCA fit queries"), cache.vectors(fit_docs, "PCA fit passages"),
                            256, out, [r["id"] for r in fit_queries], [r["id"] for r in fit_docs])
        examples = list(read_rows(args.training / "examples.jsonl"))
        docs = list(read_rows(args.training / "passages.jsonl"))
        lookup = {row["id"]: i for i, row in enumerate(docs)}
        positives = [[lookup[pid] for pid in row["positives"]] for row in examples]
        negatives = np.full((len(examples), max(len(r["negatives"]) for r in examples)), -1, dtype=np.int64)
        for i, row in enumerate(examples):
            if not row["positives"] or set(row["positives"]).intersection(row["negatives"]):
                raise ValueError(f"Invalid training candidates for {row['id']}")
            negatives[i, :len(row["negatives"])] = [lookup[pid] for pid in row["negatives"]]
        train_inputs = (torch.from_numpy(normalize(cache.vectors(examples, "train queries")) - mean),
                        torch.from_numpy(normalize(cache.vectors(docs, "train passages")) - mean),
                        positives, negatives)
        ids, passages, dev, queries = development_data(args.prepared, args.passages, cache)
        context = ids, normalize(passages), dev, normalize(queries)
        best = {}
        best_key = None
        history = {}
        results = {}
        for lr, tau in product(sorted(set(args.learning_rates)), sorted(set(args.temperatures))):
            def select(entry, weights):
                nonlocal best_key, best
                quality = rank_exact(context, (mean, weights))
                entry.update(quality)
                key = (quality["ndcg@10"], quality["recall@10"], -entry["epoch"], -lr, -tau)
                if best_key is None or key > best_key:
                    best_key = key
                    best = entry.copy()
                    save_projection(out / "learned128.npz", mean, weights)
            history[f"main-lr{lr}-tau{tau}"] = fit_trial(train_inputs, pca[:128], args.epochs, lr, tau,
                                                         True, args.seed, args.batch_size, select)
            print(f"Finished lr={lr} temperature={tau}", flush=True)

        def score(name, weights):
            results[name] = rank_exact(context, weights, out / f"{name}-k10.jsonl")

        score("raw768", None)
        for dim in (64, 128, 256):
            score(f"pca{dim}", (mean, pca[:dim]))
        score("learned128", load_projection(out / "learned128.npz"))
        for name, dim, use_hard in (("learned64", 64, True), ("learned256", 256, True),
                                    ("random128", 128, True), ("inbatch128", 128, False)):
            initial = pca[:dim]
            if name == "random128":
                generator = torch.Generator().manual_seed(args.seed)
                initial = torch.empty(dim, DIMENSION).uniform_(-DIMENSION**-0.5, DIMENSION**-0.5,
                                                                generator=generator).numpy()
            def save_last(entry, weights):
                if entry["epoch"] == best["epoch"]:
                    save_projection(out / f"{name}.npz", mean, weights)
                    score(name, (mean, weights))
                    entry.update(results[name])
            history[name] = fit_trial(train_inputs, initial, best["epoch"], best["lr"], best["temperature"],
                                      use_hard, args.seed, args.batch_size, save_last)

        paired_dev_intervals(out, results, args.seed)
        write_json(out / "summary.json", {
            "selection": "highest exact dev nDCG@10, Recall@10, earlier epoch, lower LR, lower temperature",
            "selected": best, "ablations": results, "history": history, "dev_passages": args.passages,
            "dev_queries": len(dev), "examples": len(examples), "batch_size": args.batch_size,
            "seed": args.seed, "learning_rates": args.learning_rates, "temperatures": args.temperatures,
            "max_epochs": args.epochs, "optimizer": "torch.optim.Adam (default betas, eps; no weight decay)",
            "embedding": cache.settings, "prepared_manifest_sha256": prepared_hash,
            "training_manifest_sha256": fingerprint(args.training / "manifest.json"),
            "inputs": {name: fingerprint(args.prepared / name) for name in
                       ("corpus-dev.jsonl", "eval-dev.jsonl", "pca-passages.jsonl", "pca-queries.jsonl")},
            "weights_sha256": {p.name: fingerprint(p) for p in sorted(out.glob("*.npz"))},
        })
    print(args.out / "summary.json", flush=True)
