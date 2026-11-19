"""Make a reproducible, training-only MS MARCO BM25 pool and labeled examples."""
import re
import sqlite3
from contextlib import closing
from heapq import nsmallest

from .io import (fingerprint, id_text, input_path, read_json, read_rows, run_directory,
                 sample_key, text_source, write_json, write_rows)


def prepare(args):
    source = args.prepared / "queries-train.jsonl"
    collection = input_path(args.raw, "collection.tsv", "collection.tar.gz")
    manifest = read_json(args.prepared / "manifest.json")
    if fingerprint(collection) != manifest["sources"]["collection"]["sha256"]:
        raise ValueError("Collection must match the prepared MS MARCO manifest")
    with run_directory(args.out) as out:
        labeled = (row for row in read_rows(source) if 1 in row["qrels"].values())
        queries = nsmallest(args.train_queries, labeled,
                            key=lambda row: sample_key(args.seed, "projection-query", row["id"]))
        if len(queries) != args.train_queries:
            raise ValueError("Requested more labeled training queries than available")
        positives = {pid for row in queries for pid, grade in row["qrels"].items() if grade == 1}
        pool = {pid for row in read_rows(source) for pid, grade in row["qrels"].items()
                if grade == 1 and pid not in positives}
        if not len(positives) <= args.pool_size <= len(positives) + len(pool):
            raise ValueError("BM25 pool must fit all sampled positives and fit within training passages")
        wanted = positives.union(nsmallest(args.pool_size - len(positives), pool,
                                           key=lambda pid: sample_key(args.seed, "bm25-passage", pid)))
        with text_source(collection, "collection.tsv") as stream:
            docs = {pid: text for pid, text in id_text(stream) if pid in wanted}
        if wanted != docs.keys():
            raise ValueError("Training passages missing from collection")
        ordered = sorted(docs, key=int)
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE passages USING fts5(text, tokenize='unicode61')")
            db.executemany("INSERT INTO passages(rowid,text) VALUES (?,?)",
                           ((int(pid)+1, docs[pid]) for pid in ordered))
            used = set()
            eligible = []
            no_hits = []
            for row in queries:
                known = [pid for pid, grade in row.pop("qrels").items() if grade == 1]
                tokens = dict.fromkeys(re.findall(r"[^\W_]+", row["text"].lower()))
                match = " OR ".join(f'"{token}"' for token in tokens)
                hits = db.execute("SELECT rowid FROM passages WHERE passages MATCH ? "
                                  "ORDER BY bm25(passages), rowid LIMIT ?",
                                  (match, args.hard_negatives + len(known))) if match else ()
                negatives = [str(key-1) for (key,) in hits if str(key-1) not in known][:args.hard_negatives]
                if not negatives:
                    no_hits.append(row["id"])
                    continue
                row.update(positives=sorted(known, key=int), negatives=negatives)
                used.update(known)
                used.update(negatives)
                eligible.append(row)
        if not eligible:
            raise ValueError("No sampled query retrieved a BM25 negative")
        write_rows(out / "examples.jsonl", eligible)
        write_rows(out / "passages.jsonl", ({"id": pid, "text": docs[pid]} for pid in sorted(used, key=int)))
        write_rows(out / "bm25-pool-ids.jsonl", ordered)
        write_json(out / "manifest.json", {
            "prepared_manifest_sha256": fingerprint(args.prepared / "manifest.json"),
            "collection_sha256": manifest["sources"]["collection"]["sha256"],
            "train_queries_sha256": fingerprint(source), "seed": args.seed, "sampled": len(queries),
            "examples": len(eligible), "no_lexical_negative_query_ids": no_hits,
            "pool_size": len(wanted), "encoded_passages": len(used), "known_positive_passages": len(positives),
            "hard_negatives_per_query_max": args.hard_negatives, "sqlite": sqlite3.sqlite_version,
            "bm25": "FTS5 unicode61 OR query, ascending bm25 then numeric passage ID; training positives only pool",
            "files": {name: fingerprint(out / name) for name in
                      ("examples.jsonl", "passages.jsonl", "bm25-pool-ids.jsonl")},
        })
