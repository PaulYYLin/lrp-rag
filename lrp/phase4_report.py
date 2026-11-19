"""Score frozen reader outputs and compare methods on paired questions."""
import ast
import io
import runpy
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.translate.meteor_score import single_meteor_score

from .io import fingerprint, read_json, read_rows, run_directory, write_json, write_rows

SCORER_HASH = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
WORDNET_HASH = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
METRICS = ("complete", "sentence_recall", "passage_recall", "em", "f1", "meteor")
REFERENCE = "raw768-flat"
METHODS = tuple(f"{rep}-{index}" for rep in ("raw768", "pca128", "learned128")
                for index in ("flat", "hnsw-m32"))


def keyed(path, expected):
    rows = list(read_rows(path))
    lookup = {r["id"]: r for r in rows}
    if len(lookup) != len(rows) or lookup.keys() != expected:
        raise ValueError(f"Inconsistent question set in {path}")
    return lookup


def paired(left, right, draws):
    delta = left - right
    result = {}
    for i, metric in enumerate(METRICS):
        lo, hi = np.quantile(delta[:, i][draws].mean(axis=1), [0.025, 0.975])
        result[metric] = dict(difference=float(delta[:, i].mean()), ci95=[float(lo), float(hi)])
    return result


def summarize(rows):
    if not rows:
        return dict(questions=0)
    return dict(questions=len(rows), complete_questions=sum(r["complete"] for r in rows),
                **{k: float(np.mean([r[k] for r in rows])) for k in METRICS})


def audit_retrieved_context(row, ranking, pages, question):
    context = row["context"]
    if ranking != row["passage_ids"] or len(set(ranking)) != 10:
        raise ValueError("Generation and retrieval ranks differ")
    by_page = {}
    remaining = context
    for i, pid in enumerate(ranking):
        page = pages[pid]
        heading = ("\n\n" if i else "") + page["title"] + "\n"
        section = heading + page["text"]
        prefix = remaining[:len(section)]
        if not section.startswith(prefix):
            raise ValueError("Reader context differs from the retrieved pages")
        by_page[pid] = prefix[len(heading):]
        remaining = remaining[len(section):]
    if remaining:
        raise ValueError("Extra text in reader context")
    expected = {}
    for f in question["supporting_facts"]:
        status = "unmapped"
        if f["status"] == "mapped":
            pid = f["page_id"]
            status = "not_retrieved"
            if pid in by_page:
                span = slice(f["char_start"], f["char_end"])
                status = "visible" if by_page[pid][span] == f["sentence"] else "truncated"
        expected[f["gold_title"], f["gold_sentence_index"]] = status
    if expected != {(t, i): s for t, i, s in row["facts"]}:
        raise ValueError("Saved sentence flags disagree with actual reader text")


def report(args):
    if fingerprint(args.scorer) != SCORER_HASH:
        raise ValueError("Use the pinned, unmodified HotpotQA v1 scorer")
    scorer = runpy.run_path(str(args.scorer))
    archive = (args.nltk_data / "corpora/wordnet.zip").resolve()
    if fingerprint(archive) != WORDNET_HASH:
        raise ValueError("Use the pinned English WordNet archive")
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    if Path(wordnet.root.zipfile.filename).resolve() != archive:
        raise ValueError("NLTK loaded a different WordNet corpus")
    tokenize = TreebankWordTokenizer().tokenize
    questions_path = args.prepared / "eval-hotpot.jsonl"
    qrows = list(read_rows(questions_path))
    questions = {q["id"]: q for q in qrows}
    manifest = read_json(args.prepared / "manifest.json")
    ranking_meta = read_json(args.retrieval / "manifest.json")
    if len(questions) != len(qrows) or len(qrows) != manifest["questions"]:
        raise ValueError("Duplicate or missing questions")
    if (ranking_meta["prepared_sha256"] != fingerprint(args.prepared / "manifest.json")
            or ranking_meta["questions_sha256"] != fingerprint(questions_path)):
        raise ValueError("Retrieval/prepared data mismatch")
    if fingerprint(args.prepared / "corpus.jsonl") != ranking_meta["corpus_sha256"]:
        raise ValueError("Corpus has changed since retrieval")
    if bool(args.retrieval256) != bool(args.answers256):
        raise ValueError("Supply both the supplemental retrieval and answer directories")
    if bool(args.retrieval_pca256) != bool(args.answers_pca256):
        raise ValueError("Supply both the PCA-256 retrieval and answer directories")
    sources = [(args.retrieval, args.answers, METHODS)]
    if args.retrieval256:
        sources.append((args.retrieval256, args.answers256, ("learned256-flat", "learned256-hnsw-m32")))
    if args.retrieval_pca256:
        sources.append((args.retrieval_pca256, args.answers_pca256, ("pca256-flat",)))
    rankings, inputs, manifests = {}, [], {}
    needed = set()
    for directory, answers, methods in sources:
        meta = read_json(directory / "manifest.json")
        for field in ("lock_sha256", "questions_sha256", "corpus_sha256", "prepared_sha256", "embedding"):
            if meta[field] != ranking_meta[field]:
                raise ValueError(f"Retrieval cohorts differ: {field}")
        if set(methods) != set(meta["points"]):
            raise ValueError("Unexpected retrieval conditions")
        inputs.append((answers, methods, fingerprint(directory / "manifest.json")))
        for name in methods:
            path = directory / f"{name}.jsonl"
            if fingerprint(path) != meta["rankings_sha256"][name]:
                raise ValueError(f"Ranking has changed: {name}")
            rankings[name] = keyed(path, questions.keys())
            needed.update(pid for r in rankings[name].values() for pid in r["passage_ids"])
    inputs.append((args.controls, ("question_only", "gold"), None))
    pages = {p["id"]: p for p in read_rows(args.prepared / "corpus.jsonl") if p["id"] in needed}
    gold = [dict(_id=q["id"], answer=q["answer"],
                 supporting_facts=[[f["gold_title"], f["gold_sentence_index"]] for f in q["supporting_facts"]])
            for q in qrows]
    results, verification, matrices = {}, {}, {}
    settings = None
    with run_directory(args.out) as out:
        write_json(out / "gold.json", gold)
        for directory, expected, retrieval in inputs:
            meta = read_json(directory / "manifest.json")
            if meta["questions_sha256"] != fingerprint(questions_path) or meta["retrieval_sha256"] != retrieval:
                raise ValueError("Answer inputs differ")
            if set(expected) != set(meta["conditions"]) or (settings is not None and meta["settings"] != settings):
                raise ValueError("Mismatched methods or reader settings")
            settings = meta["settings"]
            for name in expected:
                path = directory / f"{name}.jsonl"
                if fingerprint(path) != meta["rows_sha256"][name]:
                    raise ValueError(f"Answer records changed in {name}")
                lookup = keyed(path, questions.keys())
                records, errors = [], Counter()
                pred = dict(answer={}, sp={})
                for q in qrows:
                    qid = q["id"]
                    row = lookup[qid]
                    if name in rankings:
                        audit_retrieved_context(row, rankings[name][qid]["passage_ids"], pages, q)
                    facts = row["facts"]
                    keys = {(t, i) for t, i, _ in facts}
                    if keys != {(f["gold_title"], f["gold_sentence_index"]) for f in q["supporting_facts"]} or len(facts) != len(keys):
                        raise ValueError(f"Evidence denominator mismatch for {qid}")
                    visible = [[t, i] for t, i, status in facts if status == "visible"]
                    statuses = {s for _, _, s in facts}
                    complete = int(len(visible) == len(keys))
                    sentence = len(visible)/len(keys)
                    page_recall = len({t for t, _, s in facts if s in ("visible", "truncated")})/len({t for t, _ in keys})
                    if (complete, sentence, page_recall) != (row["complete"], row["sentence_recall"], row["passage_recall"]):
                        raise ValueError(f"Inconsistent evidence scores for {qid}")
                    if row["context_tokens"] > settings["context_budget"]:
                        raise ValueError("Reader context exceeds the fixed budget")
                    em = int(scorer["exact_match_score"](row["prediction"], q["answer"]))
                    f1 = scorer["f1_score"](row["prediction"], q["answer"])[0]
                    meteor = single_meteor_score(tokenize(q["answer"]), tokenize(row["prediction"]), wordnet=wordnet)
                    mapped = all(f["status"] == "mapped" for f in q["supporting_facts"])
                    if row["fully_mapped"] != mapped:
                        raise ValueError("Mapping flag mismatch")
                    records.append(dict(id=qid, prediction=row["prediction"], gold=q["answer"], complete=complete,
                                        sentence_recall=sentence, passage_recall=page_recall, em=em, f1=f1,
                                        meteor=meteor, fully_mapped=mapped, facts=facts))
                    errors["context_over_budget"] += row["before_truncation_tokens"] > settings["context_budget"]
                    for status in ("unmapped", "not_retrieved", "truncated"):
                        errors[f"at_least_one_{status}"] += status in statuses
                        errors[f"facts_{status}"] += sum(s == status for _, _, s in facts)
                    errors["complete_wrong_em"] += bool(complete and not em)
                    errors["incomplete_correct_em"] += bool(not complete and em)
                    errors["complete_correct_em"] += bool(complete and em)
                    errors["incomplete_wrong_em"] += bool(not complete and not em)
                    pred["answer"][qid], pred["sp"][qid] = row["prediction"], visible
                write_rows(out / f"{name}-scores.jsonl", records)
                prediction_file = out / f"{name}-predictions.json"
                write_json(prediction_file, pred)
                capture = io.StringIO()
                with redirect_stdout(capture):
                    scorer["eval"](str(prediction_file), str(out / "gold.json"))
                official = ast.literal_eval(capture.getvalue().strip())
                result = summarize(records)
                for k, expected_metric in (("em", "em"), ("f1", "f1"), ("sp_recall", "sentence_recall"),
                                           ("sp_em", "complete")):
                    if not np.isclose(official[k], result[expected_metric], rtol=0, atol=1e-12):
                        raise ValueError(f"Official score disagrees: {name} {k}")
                results[name] = dict(all=result, fully_mapped=summarize([r for r in records if r["fully_mapped"]]),
                                     error_counts=dict(errors))
                verification[name] = dict(questions=len(records), official=official,
                                          retrieved_contexts_audited=len(records) if name in rankings else 0)
                matrices[name] = np.array([[r[k] for k in METRICS] for r in records])
        rng = np.random.default_rng(42)
        draws = rng.integers(len(qrows), size=(10000, len(qrows)), dtype=np.int32)
        comparisons = {name: paired(values, matrices[REFERENCE], draws) for name, values in matrices.items()}
        for name in rankings:
            results[name]["passes_evidence_margin"] = comparisons[name]["complete"]["ci95"][0] >= -0.02
        learned_pca = {index: paired(matrices[f"learned128-{index}"], matrices[f"pca128-{index}"], draws)
                       for index in ("flat", "hnsw-m32")}
        within_rep = {rep: paired(matrices[f"{rep}-hnsw-m32"], matrices[f"{rep}-flat"], draws)
                      for rep in ("raw768", "learned128", "pca128")}
        dimension = {}
        if args.retrieval256:
            for index in ("flat", "hnsw-m32"):
                dimension[index] = paired(matrices[f"learned256-{index}"], matrices[f"learned128-{index}"], draws)
            within_rep["learned256"] = paired(matrices["learned256-hnsw-m32"], matrices["learned256-flat"], draws)
            manifests.update(retrieval256=fingerprint(args.retrieval256 / "manifest.json"),
                             answers256=fingerprint(args.answers256 / "manifest.json"))
        extra = {}
        if args.retrieval_pca256:
            manifests.update(retrieval_pca256=fingerprint(args.retrieval_pca256 / "manifest.json"),
                             answers_pca256=fingerprint(args.answers_pca256 / "manifest.json"))
            pca256 = matrices["pca256-flat"]
            extra["pca256_minus_128"] = paired(pca256, matrices["pca128-flat"], draws)
            if args.retrieval256:
                extra["learned256_minus_pca256"] = paired(matrices["learned256-flat"], pca256, draws)
        write_json(out / "summary.json", dict(conditions=results, learned256_minus_128=dimension, reference=REFERENCE, comparisons=comparisons,
                   learned_minus_pca=learned_pca, hnsw_minus_flat=within_rep,
                   bootstrap=dict(seed=42, resamples=10000, unit="question",
                   interval="paired percentile 95%"), evidence_margin=-0.02, scorer_sha256=SCORER_HASH,
                   meteor=dict(implementation="nltk.translate.meteor_score.single_meteor_score",
                               nltk=nltk.__version__, tokenizer="TreebankWordTokenizer", lowercase=True,
                               stemmer="PorterStemmer", wordnet=wordnet.get_version(), wordnet_sha256=WORDNET_HASH,
                               alpha=0.9, beta=3.0, gamma=0.5, reference="single original answer"),
                   input_manifests=manifests | {k: fingerprint(p / "manifest.json") for k, p in (
                       ("controls", args.controls), ("answers", args.answers), ("retrieval", args.retrieval),
                       ("prepared", args.prepared))}, **extra))
        write_json(out / "verification.json", verification)
