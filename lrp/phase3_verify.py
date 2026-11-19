"""Verify corpus membership and individual test scores with a supplied NIST trec_eval."""
import subprocess
import tempfile
from itertools import islice
from pathlib import Path

from .io import fingerprint, read_rows


def verify(directory, prepared, summary, scorer):
    ids = list(read_rows(directory / "passage-ids.jsonl"))
    expected = [r["id"] for r in islice(read_rows(prepared / "corpus-trec.jsonl"), summary["passages"])]
    if ids != expected or len(set(ids)) != summary["passages"]:
        raise ValueError("Index IDs do not match the nested passage corpus")
    corpus = set(ids)
    qrels = {r["id"]: r["qrels"] for year in ("trec2019", "trec2020")
             for r in read_rows(prepared / f"eval-{year}.jsonl")}
    for name, cost in summary["indexes"].items():
        if (directory / f"{name}.faiss").stat().st_size != cost["index_bytes"]:
            raise ValueError(f"Index size changed for {name}")
    if fingerprint(directory / "build.json") != summary["build_sha256"]:
        raise ValueError("The measured indexes and build summary do not match")
    names = sorted(p["name"] for p in summary["configurations"])
    for order in summary["order"]:
        if sorted(order["queries"]) != sorted(qrels) or sorted(order["points"]) != names:
            raise ValueError("Incomplete randomized timing round")
    count = checks = 0
    max_error = 0
    with tempfile.TemporaryDirectory() as tmp:
        truth, run = Path(tmp) / "truth", Path(tmp) / "run"
        truth.write_text("".join(f"{qid} 0 {pid} {grade}\n" for qid, rels in qrels.items()
                                 for pid, grade in rels.items()))
        for point in summary["configurations"]:
            k, name = point["k"], point["name"]
            measures = ["-m", f"recall.{k}"]
            if k == 10:
                measures += ["-m", "ndcg_cut.10"]
            for rankings in (directory / f"{name}.jsonl", *(
                    directory / "measurement" / f"{name}-batch{b}.jsonl" for b in (1, 32))):
                records = list(read_rows(rankings))
                metrics = {r["id"]: r["metrics"] for r in records}
                if len(records) != len(qrels) or metrics.keys() != qrels.keys():
                    raise ValueError(f"Incomplete query results in {rankings}")
                with run.open("w") as stream:
                    for row in records:
                        ranked = row["passage_ids"]
                        if len(ranked) != k or len(set(ranked)) != k or not corpus.issuperset(ranked):
                            raise ValueError(f"Invalid result list in {rankings}")
                        for pos, pid in enumerate(ranked):
                            stream.write(f"{row['id']} Q0 {pid} {pos+1} {k-pos} lrp\n")
                output = subprocess.check_output([str(scorer), "-q", "-l", "2", *measures,
                                                  str(truth), str(run)], text=True)
                query_scores = 0
                for line in output.splitlines():
                    field, qid, value = line.split()
                    if qid == "all":
                        continue
                    metric = "ndcg@10" if field == "ndcg_cut_10" else f"recall@{k}"
                    error = abs(metrics[qid][metric]-float(value))
                    if error > 0.00005001:
                        raise ValueError(f"{rankings}: {qid} {metric} disagrees with trec_eval")
                    max_error = max(max_error, error)
                    query_scores += 1
                if query_scores != len(qrels) * (2 if k == 10 else 1):
                    raise ValueError("Incomplete external scoring output")
                count += len(records)
                checks += query_scores
    return dict(rankings_checked=count, scores_checked=checks, maximum_rounding_error=max_error,
                indexes_checked=len(summary["indexes"]), configurations_checked=len(names))
