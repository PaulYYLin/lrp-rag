from collections import Counter, defaultdict
from heapq import nsmallest

from .io import (id_text, input_path, normalized_query, sample_key, source_files,
                 text_source, write_json, write_rows)

MSMARCO_PASSAGES = 8_841_823
SPLITS = ("train", "dev", "trec2019", "trec2020")


def queries(path, alias=None):
    with text_source(path, alias) as stream:
        rows = {}
        for identifier, text in id_text(stream):
            if identifier in rows:
                raise ValueError(f"Duplicate query ID {identifier} in {path}")
            if not normalized_query(text):
                raise ValueError(f"Empty normalized query {identifier} in {path}")
            rows[identifier] = text
    return rows


def qrels(path, allowed_grades):
    result = defaultdict(dict)
    duplicate_rows = 0
    with text_source(path) as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 4 or fields[1] not in ("0", "Q0"):
                raise ValueError(f"Invalid qrel row {number} in {path}")
            qid, _, pid, grade = fields
            grade = int(grade)
            if grade not in allowed_grades:
                raise ValueError(f"Invalid judgment for {qid}/{pid} in {path}")
            if pid in result[qid]:
                if result[qid][pid] != grade:
                    raise ValueError(f"Conflicting judgments for {qid}/{pid} in {path}")
                duplicate_rows += 1
            result[qid][pid] = grade
    return result, duplicate_rows


def filter_split(rows, *higher):
    counts = Counter({"id_only": 0, "text_only": 0, "id_and_text": 0})
    other_ids = {identifier for split in higher for identifier in split}
    other_texts = {normalized_query(text) for split in higher for text in split.values()}
    kept = {}
    excluded = []
    for identifier, text in rows.items():
        id_match = identifier in other_ids
        text_match = normalized_query(text) in other_texts
        if id_match or text_match:
            counts["id_and_text" if id_match and text_match
                   else "id_only" if id_match else "text_only"] += 1
            excluded.append({"id": identifier, "id_match": id_match, "normalized_text_match": text_match})
        else:
            kept[identifier] = text
    return kept, dict(counts), excluded


def eval_rows(rows, labels, identifiers):
    return ({"id": qid, "text": rows[qid], "qrels": labels[qid]} for qid in identifiers)


def passage_ids(labels):
    return {pid for judgments in labels for pid in judgments}


def sample_corpora(path, forced, size, seed, training_ids, expected_count):
    wanted = set(training_ids).union(*forced.values())
    retained = {}
    seen = bytearray(expected_count)
    count = 0

    def background():
        nonlocal count
        with text_source(path, "collection.tsv") as stream:
            for identifier, text in id_text(stream):
                pid = int(identifier)
                if not 0 <= pid < expected_count or identifier != str(pid) or seen[pid]:
                    raise ValueError(f"Invalid or duplicate MS MARCO passage ID {identifier}")
                seen[pid] = 1
                count += 1
                if identifier in wanted:
                    retained[identifier] = text
                else:
                    yield sample_key(seed, "marco-passage", identifier), text

    common = nsmallest(size, background())
    if count != expected_count:
        raise ValueError(f"Expected {expected_count:,} MS MARCO passages, found {count:,}")
    return {
        name: (
            [{"id": pid, "text": retained[pid]} for pid in sorted(pids, key=int)]
            + [{"id": pid, "text": text} for (_, pid), text in nsmallest(
                size - len(pids), common + [
                    (sample_key(seed, "marco-passage", pid), text)
                    for pid, text in retained.items() if pid not in pids
                ]
            )]
        )
        for name, pids in forced.items()
    }, retained


def split_stats(rows, labels):
    return {
        "queries": len(rows),
        "judged_queries": len(labels),
        "unique_judged_passages": len(passage_ids(labels.values())),
        "judgments_by_grade": dict(sorted(Counter(
            str(grade) for judgments in labels.values() for grade in judgments.values()
        ).items())),
    }


def prepare(raw, out, sizes, dev_limit, pca_count, seed, expected_count=MSMARCO_PASSAGES):
    sources = {}
    rows = {}
    labels = {}
    duplicate_rows = {}
    for name in SPLITS:
        if name.startswith("trec"):
            year = name[-4:]
            sources[f"queries_{name}"] = input_path(
                raw, f"msmarco-test{year}-queries.tsv.gz", f"msmarco-test{year}-queries.tsv"
            )
            sources[f"qrels_{name}"] = input_path(raw, f"{year}qrels-pass.txt")
            rows[name] = queries(sources[f"queries_{name}"])
            labels[name], duplicate_rows[name] = qrels(sources[f"qrels_{name}"], range(4))
        else:
            member = f"queries.{name}.tsv"
            sources[f"queries_{name}"] = input_path(raw, member, "queries.tar.gz")
            sources[f"qrels_{name}"] = input_path(raw, f"qrels.{name}.tsv")
            rows[name] = queries(sources[f"queries_{name}"], member)
            labels[name], duplicate_rows[name] = qrels(sources[f"qrels_{name}"], (0, 1))

        missing = labels[name].keys() - rows[name].keys()
        if missing:
            raise ValueError(f"{name}: qrels reference {len(missing)} missing query IDs")
        for judgments in labels[name].values():
            for pid in judgments:
                if not 0 <= int(pid) < expected_count or pid != str(int(pid)):
                    raise ValueError(f"{name}: invalid MS MARCO passage ID {pid}")

    original = {name: split_stats(rows[name], labels[name]) for name in SPLITS}
    test_rows = rows["trec2019"], rows["trec2020"]
    rows["dev"], dev_removed, dev_excluded = filter_split(rows["dev"], *test_rows)
    rows["train"], train_removed, train_excluded = filter_split(rows["train"], rows["dev"], *test_rows)
    for name in ("train", "dev"):
        labels[name] = {qid: grades for qid, grades in labels[name].items() if qid in rows[name]}

    training_queries = [qid for qid in labels["train"] if 1 in labels["train"][qid].values()]
    training_passages = {
        pid for grades in labels["train"].values() for pid, grade in grades.items() if grade == 1
    }
    if min(len(training_queries), len(training_passages)) < pca_count:
        raise ValueError(f"Insufficient training data for {pca_count} samples per PCA source")
    fit_qids = nsmallest(pca_count, training_queries, key=lambda qid: sample_key(seed, "pca-query", qid))
    fit_pids = nsmallest(pca_count, training_passages, key=lambda pid: sample_key(seed, "pca-passage", pid))

    if not labels["dev"]:
        raise ValueError("No judged development queries remain")
    dev_qids = nsmallest(dev_limit, labels["dev"], key=lambda qid: sample_key(seed, "dev-query", qid))
    forced = {
        "dev": passage_ids(labels["dev"][qid] for qid in dev_qids),
        "trec": passage_ids(labels["trec2019"].values()) | passage_ids(labels["trec2020"].values()),
    }
    if min(sizes) < max(map(len, forced.values())) or max(sizes) > expected_count:
        raise ValueError(f"Invalid sizes {sizes} for forced counts "
                         f"{ {key: len(value) for key, value in forced.items()} } "
                         f"and collection size {expected_count}")

    sources["collection"] = input_path(raw, "collection.tsv", "collection.tar.gz")
    corpora, retained = sample_corpora(sources["collection"], forced, max(sizes),
                                      seed, fit_pids, expected_count)
    corpus_stats = {}
    for name, corpus in corpora.items():
        write_rows(out / f"corpus-{name}.jsonl", corpus)
        corpus_stats[name] = {
            str(size): {"judged_passages": len(forced[name]),
                        "background_passages": size - len(forced[name]),
                        "judged_fraction": len(forced[name]) / size}
            for size in sizes
        }

    for name, identifiers in (("dev", dev_qids), ("trec2019", sorted(labels["trec2019"], key=int)),
                              ("trec2020", sorted(labels["trec2020"], key=int))):
        write_rows(out / f"eval-{name}.jsonl", eval_rows(rows[name], labels[name], identifiers))
    for name in SPLITS:
        write_rows(out / f"queries-{name}.jsonl",
                   ({"id": qid, "text": text, "qrels": labels[name].get(qid, {})}
                    for qid, text in rows[name].items()))
    write_rows(out / "excluded-queries.jsonl",
               (dict(row, split=name) for name, excluded in (("dev", dev_excluded), ("train", train_excluded))
                for row in excluded))
    write_rows(out / "pca-queries.jsonl", ({"id": qid, "text": rows["train"][qid]} for qid in fit_qids))
    write_rows(out / "pca-passages.jsonl", ({"id": pid, "text": retained[pid]} for pid in fit_pids))

    source_metadata = source_files(sources)
    for name in ("train", "dev"):
        if sources[f"queries_{name}"].name.endswith(".tar.gz"):
            source_metadata[f"queries_{name}"]["member"] = f"queries.{name}.tsv"
    if sources["collection"].name.endswith(".tar.gz"):
        source_metadata["collection"]["member"] = "collection.tsv"

    write_json(out / "manifest.json", {
        "dataset": "MS MARCO Passage Ranking v1 / TREC DL Passage Ranking 2019 and 2020",
        "sources": source_metadata, "seed": seed, "collection_passages": expected_count,
        "sampling": "forced ascending numeric ID, then BLAKE2b-64(seed, domain, ID) and ID",
        "sizes": sizes, "corpora": corpus_stats, "pca_samples_per_source": pca_count,
        "original_splits": original,
        "identical_duplicate_qrel_rows_deduplicated": duplicate_rows,
        "decontaminated_splits": {name: split_stats(rows[name], labels[name]) for name in SPLITS},
        "removed_from_dev": dev_removed, "removed_from_train": train_removed,
        "trec_shared_query_ids": sorted(rows["trec2019"].keys() & rows["trec2020"].keys()),
        "dev_evaluation": split_stats(dev_qids, {qid: labels["dev"][qid] for qid in dev_qids}),
        "all_trec_judgments_retained_in_each_test_corpus": True,
        "all_qrel_passage_ids_validated": True,
    })
