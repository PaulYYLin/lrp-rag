import argparse
from itertools import islice
from pathlib import Path

from .io import fingerprint, read_json, read_rows, run_directory, write_json


def positive(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return count


def prepare_marco(args):
    from .marco import prepare

    with run_directory(args.out) as out:
        prepare(args.raw, out, sorted(set(args.sizes)), args.dev_queries, args.pca_samples,
                args.seed, args.collection_count)
    print(args.out / "manifest.json")


def prepare_hotpot(args):
    from .hotpot import prepare

    with run_directory(args.out) as out:
        prepare(args.fullwiki, args.distractor, args.wiki, out, args.questions,
                args.passages, args.seed, args.collection_count)
    print(args.out / "manifest.json")


def evaluate(args):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from .benchmark import fit_pca, run
    from .embedding import EmbeddingCache

    prepared = args.prepared
    manifest = read_json(prepared / "manifest.json")
    if args.split == "hotpot":
        corpus_path = prepared / "corpus.jsonl"
        available = manifest["corpus_passages"]
        splits = ("hotpot",)
        if args.passages != available or args.pca_source is None:
            raise ValueError(f"HotpotQA needs --passages {available} and --pca-source pointing to MS MARCO")
    else:
        corpus_path = prepared / f"corpus-{args.split}.jsonl"
        available = max(manifest["sizes"])
        splits = ("dev",) if args.split == "dev" else ("trec2019", "trec2020")
        if args.passages < min(manifest["sizes"]):
            raise ValueError(f"Choose at least {min(manifest['sizes'])} passages from the nested corpus")
    if args.passages > available:
        raise ValueError(f"Only {available} passages were prepared")
    corpus = list(islice(read_rows(corpus_path), args.passages))
    if len(corpus) != args.passages:
        raise ValueError(f"Missing prepared corpus rows in {corpus_path}")

    groups = [(name, list(read_rows(prepared / f"eval-{name}.jsonl"))) for name in splits]
    if args.split != "hotpot":
        ids = {row["id"] for row in corpus}
        for name, rows in groups:
            missing = {pid for row in rows for pid in row["qrels"] if pid not in ids}
            if missing:
                raise ValueError(f"{name}: {len(missing)} judged passages are missing from the benchmark corpus")
    pca_source = args.pca_source or prepared
    query_fit_path = pca_source / "pca-queries.jsonl"
    passage_fit_path = pca_source / "pca-passages.jsonl"
    train_queries = list(read_rows(query_fit_path))
    train_passages = list(read_rows(passage_fit_path))

    links = sorted(set(args.links or ([16, 32, 48] if args.passages == 100_000 and args.split != "hotpot" else [32])))
    ef_values = sorted(set(args.ef_search))
    if max(ef_values) < 100:
        raise ValueError("Supply at least one efSearch >= 100 for Recall@100")
    with run_directory(args.out) as out, threadpool_limits(limits=args.threads, user_api="blas"):
        cache = EmbeddingCache(args.cache, args.device, args.batch_size, args.max_length,
                               args.threads, args.model_dir)
        train_query_vectors = cache.vectors(train_queries, "PCA train queries")
        train_passage_vectors = cache.vectors(train_passages, "PCA train passages")
        pca = fit_pca(train_query_vectors, train_passage_vectors, args.dimensions, out,
                      [row["id"] for row in train_queries], [row["id"] for row in train_passages])
        query_vectors = np.concatenate([cache.vectors(rows, name) for name, rows in groups])
        documents = cache.vectors(corpus, f"Corpus {args.split}")
        run(out, corpus, documents, groups, query_vectors, pca, args.seed, args.threads,
            args.repetitions, links, ef_values=ef_values)
        write_json(out / "manifest.json", {
            "split": args.split, "passages": args.passages, "embedding": cache.settings,
            "cache": cache.path.name, "prepared_manifest_sha256": fingerprint(prepared / "manifest.json"),
            "pca_queries_sha256": fingerprint(query_fit_path),
            "pca_passages_sha256": fingerprint(passage_fit_path),
            "corpus_file": corpus_path.name, "corpus_sha256": fingerprint(corpus_path),
            "evaluation_sha256": {name: fingerprint(prepared / f"eval-{name}.jsonl") for name in splits},
        })
    print(args.out / "summary.json")


def main():
    parser = argparse.ArgumentParser(description="LRP data preparation, projection training and retrieval experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    marco = commands.add_parser("prepare-marco", help="Split, sample, and audit MS MARCO and TREC DL")
    marco.add_argument("--raw", type=Path, default=Path("data/raw"))
    marco.add_argument("--out", type=Path, default=Path("artifacts/marco"))
    marco.add_argument("--sizes", type=positive, nargs="+", default=[30_000, 100_000, 500_000])
    marco.add_argument("--dev-queries", type=positive, default=500)
    marco.add_argument("--pca-samples", type=positive, default=2048)
    marco.add_argument("--collection-count", type=positive, default=8_841_823)
    marco.add_argument("--seed", type=int, default=42)
    marco.set_defaults(action=prepare_marco)

    hotpot = commands.add_parser("prepare-hotpot", help="Audit and sample official HotpotQA introductions")
    hotpot.add_argument("--fullwiki", type=Path, required=True)
    hotpot.add_argument("--distractor", type=Path, required=True)
    hotpot.add_argument("--wiki", type=Path, required=True)
    hotpot.add_argument("--out", type=Path, default=Path("artifacts/hotpot"))
    hotpot.add_argument("--questions", type=positive, default=500)
    hotpot.add_argument("--passages", type=positive, default=100_000)
    hotpot.add_argument("--collection-count", type=positive, default=5_233_329)
    hotpot.add_argument("--seed", type=int, default=42)
    hotpot.set_defaults(action=prepare_hotpot)

    baseline = commands.add_parser("baseline", help="Cache Contriever vectors and run the four baseline configurations")
    baseline.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
    baseline.add_argument("--pca-source", type=Path, help="MS MARCO training sample directory for HotpotQA")
    baseline.add_argument("--split", choices=("dev", "trec", "hotpot"), default="dev")
    baseline.add_argument("--passages", type=positive, default=30_000)
    baseline.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    baseline.add_argument("--model-dir", type=Path)
    baseline.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    baseline.add_argument("--max-length", type=positive, default=128)
    baseline.add_argument("--batch-size", type=positive, default=64)
    baseline.add_argument("--threads", type=positive, default=1)
    baseline.add_argument("--dimensions", type=positive, choices=(64, 128, 256), default=128)
    baseline.add_argument("--repetitions", type=positive, default=5)
    baseline.add_argument("--links", type=positive, nargs="+", choices=(16, 32, 48))
    ef_grid = (10, 16, 32, 64, 128, 256, 512)
    baseline.add_argument("--ef-search", type=positive, nargs="+", choices=ef_grid, default=ef_grid)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.set_defaults(action=evaluate)

    from .training_data import prepare as prepare_projection

    train_data = commands.add_parser("prepare-projection", help="Mine BM25 negatives from training passages")
    train_data.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
    train_data.add_argument("--raw", type=Path, default=Path("data/raw"))
    train_data.add_argument("--out", type=Path, default=Path("artifacts/projection-data"))
    train_data.add_argument("--train-queries", type=positive, default=8192)
    train_data.add_argument("--pool-size", type=positive, default=32768)
    train_data.add_argument("--hard-negatives", type=positive, default=4)
    train_data.add_argument("--seed", type=int, default=42)
    train_data.set_defaults(action=prepare_projection)

    training = commands.add_parser("train-projection", help="Tune the main projection and run matched ablations")
    training.add_argument("--training", type=Path, default=Path("artifacts/projection-data"))
    training.add_argument("--out", type=Path, default=Path("artifacts/projection"))
    training.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
    training.add_argument("--model-dir", type=Path)
    training.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    training.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    training.add_argument("--passages", type=positive, default=30000)
    training.add_argument("--batch-size", type=positive, default=128)
    training.add_argument("--epochs", type=positive, default=6)
    training.add_argument("--learning-rates", type=float, nargs="+", default=[0.0003, 0.001])
    training.add_argument("--temperatures", type=float, nargs="+", default=[0.03, 0.07])
    training.add_argument("--seed", type=int, default=42)

    def train_projection(args):
        from .projection import train
        train(args)

    training.set_defaults(action=train_projection)

    calibration = commands.add_parser("calibrate-projection", help="Measure a development corpus and select HNSW points")
    calibration.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
    calibration.add_argument("--projection", type=Path, default=Path("artifacts/projection"))
    calibration.add_argument("--passages", type=positive, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    calibration.add_argument("--model-dir", type=Path)
    calibration.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    calibration.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    calibration.add_argument("--repetitions", type=positive, default=5)
    calibration.add_argument("--seed", type=int, default=42)

    def calibrate_projection(args):
        from .calibration import calibrate
        calibrate(args)

    calibration.set_defaults(action=calibrate_projection)

    lock = commands.add_parser("lock-projection", help="Freeze checkpoints and size-specific test settings")
    lock.add_argument("--projection", type=Path, default=Path("artifacts/projection"))
    lock.add_argument("--calibration", type=Path, nargs="+", required=True)
    lock.add_argument("--out", type=Path, default=Path("artifacts/phase2-lock.json"))

    def lock_projection(args):
        from .calibration import lock
        lock(args)

    lock.set_defaults(action=lock_projection)

    def phase3_action(args):
        from . import phase3
        action = phase3.build if args.command == "phase3-build" else phase3.measure
        action(args)

    for command in ("phase3-build", "phase3-measure"):
        experiment = commands.add_parser(command, help="Build or time the locked TREC test grid")
        experiment.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
        experiment.add_argument("--lock", type=Path, default=Path("artifacts/phase2-lock.json"))
        experiment.add_argument("--cache", type=Path, default=Path("data/embeddings"))
        experiment.add_argument("--model-dir", type=Path)
        experiment.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
        experiment.add_argument("--passages", type=positive, required=True)
        experiment.add_argument("--out", type=Path, required=True)
        experiment.set_defaults(action=phase3_action)

    report = commands.add_parser("phase3-report", help="Analyze paired test outcomes and plot curves")
    report.add_argument("--results", type=Path, default=Path("artifacts"))
    report.add_argument("--lock", type=Path, default=Path("artifacts/phase2-lock.json"))
    report.add_argument("--out", type=Path, default=Path("artifacts/phase3-report"))

    def phase3_report(args):
        from .phase3_analysis import report
        report(args)

    report.add_argument("--trec-eval", type=Path, help="Cross-check each query with a local trec_eval binary")
    report.add_argument("--prepared", type=Path, default=Path("artifacts/marco"))
    report.set_defaults(action=phase3_report)

    from .phase4_cli import register
    register(commands)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
