"""Search the prepared HotpotQA corpus at the matching development operating points."""
from time import perf_counter

import faiss
from threadpoolctl import threadpool_limits

from .benchmark import create_index
from .io import fingerprint, read_json, read_rows, run_directory, write_json, write_rows
from .phase3 import locked_encoder


def retrieve(args):
    source = args.prepared
    manifest = read_json(source / "manifest.json")
    args.passages = manifest["corpus_passages"]
    lock, cache, transforms, training = locked_encoder(args)
    locked = lock["sizes"][str(args.passages)]
    points = {kind: locked[kind] for kind in ("flat_k10", "k10")}
    extra = {}
    if args.pca256:
        from .phase4_256 import pca256
        transforms = {"pca256": pca256(lock)}
        points = {"flat_k10": {"pca256": dict(m=None, k=10, ef_search=None)}}
        extra = {"weights": dict(lock["pca"], components="all 256 rows")}
    elif args.calibration256:
        from .calibration import choose
        from .phase4_256 import learned256
        calibration = read_json(args.calibration256)
        preprocess, weights = learned256(lock, training)
        if (calibration["source_lock_sha256"] != fingerprint(args.lock) or calibration["weights"] != weights
                or calibration["split"] != "MS MARCO dev" or calibration["passages"] != args.passages):
            raise ValueError("Supplemental calibration does not match the encoder lock and corpus size")
        reference = dict(points["flat_k10"]["raw768"], representation="raw768")
        selected = choose([reference, *calibration["configurations"]], 10, ["learned256"])
        if selected["learned256"] != calibration["selected"]:
            raise ValueError("Supplemental HNSW choice differs from the selection rule")
        transforms = {"learned256": preprocess}
        points = {"flat_k10": {"learned256": dict(m=None, k=10, ef_search=None)}, "k10": selected}
        extra = dict(calibration256_sha256=fingerprint(args.calibration256), weights=weights)
    corpus = list(read_rows(source / "corpus.jsonl"))
    ids = [page["id"] for page in corpus]
    questions = list(read_rows(source / "eval-hotpot.jsonl"))
    if len(set(ids)) != len(ids) or len(ids) != args.passages:
        raise ValueError("Invalid HotpotQA corpus IDs or size")
    if len({q["id"] for q in questions}) != len(questions) or len(questions) != manifest["questions"]:
        raise ValueError("Invalid HotpotQA question IDs or count")
    costs = {}
    with run_directory(args.out) as out, threadpool_limits(limits=1, user_api="blas"):
        docs = cache.vectors(corpus, "Hotpot passages")
        queries = cache.vectors(questions, "Hotpot queries")
        del corpus
        for rep, transform in transforms.items():
            vectors, query_vectors = transform(docs), transform(queries)
            for methods in points.values():
                point = methods[rep]
                faiss.omp_set_num_threads(lock["build_threads"])
                index, name, cost = create_index(vectors, rep, out, lock["index_seed"], point["m"])
                if point["m"]:
                    index.hnsw.efSearch = point["ef_search"]
                faiss.omp_set_num_threads(lock["search_threads"])
                start = perf_counter()
                scores, positions = index.search(query_vectors, point["k"])
                cost.update(search_seconds=perf_counter()-start, ef_search=point["ef_search"], k=point["k"])
                if any(min(row) < 0 or len(set(row)) != 10 for row in positions):
                    raise ValueError("Incomplete top-10 ranking")
                write_rows(out / f"{name}.jsonl", (
                    dict(id=q["id"], passage_ids=[ids[i] for i in rank], scores=score.tolist())
                    for q, rank, score in zip(questions, positions, scores, strict=True)))
                costs[name] = cost
                print(name, flush=True)
                del index
        write_rows(out / "passage-ids.jsonl", ids)
        write_json(out / "manifest.json", dict(
            lock_sha256=fingerprint(args.lock), prepared_sha256=fingerprint(source / "manifest.json"),
            corpus_sha256=fingerprint(source / "corpus.jsonl"), questions_sha256=fingerprint(source / "eval-hotpot.jsonl"),
            rankings_sha256={name: fingerprint(out / f"{name}.jsonl") for name in costs},
            points=costs, embedding=cache.settings, faiss=faiss.__version__, passages=len(ids),
            documents_encoded_as="body text only", questions=len(questions), **extra))
