import bz2
import json
import tarfile
import unicodedata
from collections import Counter, defaultdict
from heapq import nsmallest
from pathlib import Path

import ijson

from .io import sample_key, source_files, write_json, write_rows

def sentences_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def title_key(title):
    return sentences_key(title).casefold()


def json_array(path):
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def file_records(path):
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def wiki_records(path):
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and not child.name.startswith(".") and child.suffix in (".bz2", ".jsonl"):
                yield from file_records(child)
    elif path.name.endswith(".tar.bz2"):
        with tarfile.open(path, "r|bz2") as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name.startswith("."):
                    continue
                with archive.extractfile(member) as raw:
                    if name.endswith(".bz2"):
                        with bz2.BZ2File(raw) as stream:
                            yield from (json.loads(line) for line in stream if line.strip())
                    else:
                        yield from (json.loads(line) for line in raw if line.strip())
    else:
        yield from file_records(path)


def facts(row):
    pairs = {}
    for title, index in row["supporting_facts"]:
        if not title or type(index) is not int or index < 0:
            raise ValueError(f"Invalid support annotation in {row['_id']}")
        pairs[title, index] = None
    if not pairs:
        raise ValueError(f"No support annotations in {row['_id']}")
    return pairs.keys()


def prepare_questions(fullwiki, distractor, count, seed):
    seen = set()

    def candidates():
        for row in json_array(fullwiki):
            identifier = row["_id"]
            if identifier in seen:
                raise ValueError(f"Duplicate HotpotQA ID {identifier}")
            seen.add(identifier)
            yield row

    selected = nsmallest(count, candidates(), key=lambda row: sample_key(seed, "hotpot-question", row["_id"]))
    if len(selected) != count:
        raise ValueError(f"Requested {count} HotpotQA questions, found {len(selected)}")
    selected_ids = {row["_id"] for row in selected}
    originals = {}
    for row in json_array(distractor):
        identifier = row["_id"]
        if identifier in selected_ids:
            if identifier in originals:
                raise ValueError(f"Duplicate distractor ID {identifier}")
            originals[identifier] = row

    pending = []
    for row in selected:
        source = originals.get(row["_id"])
        if source and any(source[field] != row[field] for field in ("question", "answer")):
            raise ValueError(f"Question or answer disagreement for {row['_id']}")
        annotated = list(facts(row))
        if source and set(facts(source)) != set(annotated):
            raise ValueError(f"Supporting facts disagree for {row['_id']}")
        context = defaultdict(list)
        if source:
            for title, sentences in source["context"]:
                context[title].append(sentences)
        evidence = []
        for title, index in annotated:
            matches = {sentences[index] for sentences in context[title]
                       if index < len(sentences) and sentences_key(sentences[index])}
            gold = next(iter(matches)) if len(matches) == 1 else None
            evidence.append({"gold_title": title, "gold_sentence_index": index, "gold_sentence": gold})
        pending.append({"id": row["_id"], "text": row["question"], "answer": row["answer"],
                        "supporting_facts": evidence})
    return pending, len(seen)


def map_fact(fact, candidates):
    if not candidates:
        return {"status": "title_missing"}
    exact_title = [page for page in candidates if page["title"] == fact["gold_title"]]
    candidates = exact_title or candidates
    index = fact["gold_sentence_index"]
    at_index = [page for page in candidates if index < len(page["sentences"])]
    if not at_index:
        return {"status": "sentence_index_missing"}
    gold = fact["gold_sentence"]
    if gold is None:
        return {"status": "gold_text_unavailable"}
    match = [page for page in at_index if sentences_key(page["sentences"][index]) == sentences_key(gold)]
    if not match:
        return {"status": "sentence_text_mismatch",
                "candidates": [{"page_id": page["id"], "sentence": page["sentences"][index]} for page in at_index]}
    if len(match) > 1:
        return {"status": "ambiguous_match", "candidate_page_ids": [page["id"] for page in match]}
    page = match[0]
    start = sum(map(len, page["sentences"][:index]))
    return {"status": "mapped", "page_id": page["id"], "page_title": page["title"],
            "sentence_index": index, "sentence": page["sentences"][index],
            "char_start": start, "char_end": start + len(page["sentences"][index])}


def prepare(fullwiki, distractor, wiki, out, questions, size, seed, expected_count):
    if size > expected_count:
        raise ValueError(f"Requested {size:,} introductions from a {expected_count:,}-passage collection")
    pending, available = prepare_questions(fullwiki, distractor, questions, seed)
    titles = {title_key(fact["gold_title"]) for row in pending for fact in row["supporting_facts"]}
    candidates = defaultdict(list)
    count = 0

    def background():
        nonlocal count
        for page in wiki_records(wiki):
            count += 1
            title = page["title"]
            identifier = str(page["id"])
            sentences = page["text"]
            if not isinstance(sentences, list) or not all(isinstance(sentence, str) for sentence in sentences):
                raise ValueError(f"Expected introductory sentences in plaintext text field for page {identifier}")
            if title_key(title) in titles:
                candidates[title_key(title)].append({"id": identifier, "title": title, "sentences": sentences})
            else:
                yield sample_key(seed, "hotpot-passage", identifier), title, sentences

    if size == expected_count:
        # A complete corpus can be emitted in source order without keeping it in RAM.
        from .io import atomic_file

        with atomic_file(out / "corpus.jsonl") as stream:
            for (_, identifier), title, sentences in background():
                row = {"id": identifier, "title": title, "text": "".join(sentences), "sentences": sentences}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            for pages in candidates.values():
                for page in pages:
                    row = dict(page, text="".join(page["sentences"]))
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        sampled = []
    else:
        sampled = nsmallest(size, background())

    if count != expected_count:
        raise ValueError(f"Expected {expected_count:,} Wikipedia introductions, found {count:,}")

    statuses = Counter()
    forced = set()
    fully_mapped = 0
    for row in pending:
        for fact in row["supporting_facts"]:
            fact.update(map_fact(fact, candidates[title_key(fact["gold_title"])]))
            statuses[fact["status"]] += 1
            if fact["status"] == "mapped":
                forced.add(fact["page_id"])
        fully_mapped += all(fact["status"] == "mapped" for fact in row["supporting_facts"])
    pages = {page["id"]: page for matches in candidates.values() for page in matches}
    if size < len(forced):
        raise ValueError(f"Corpus size {size} cannot retain the {len(forced)} mapped supporting pages")
    if size < count:
        other = sampled + [(sample_key(seed, "hotpot-passage", pid), page["title"], page["sentences"])
                           for pid, page in pages.items() if pid not in forced]
        corpus = [dict(pages[pid], text="".join(pages[pid]["sentences"])) for pid in sorted(forced, key=int)]
        corpus.extend({"id": pid, "title": title, "text": "".join(sentences), "sentences": sentences}
                      for (_, pid), title, sentences in nsmallest(size - len(forced), other))
        if len({row["id"] for row in corpus}) != size:
            raise ValueError("The HotpotQA sample contains duplicate or missing Wikipedia IDs")
        write_rows(out / "corpus.jsonl", corpus)
    write_rows(out / "eval-hotpot.jsonl", pending)

    sources = {"fullwiki": fullwiki, "distractor_gold_text": distractor}
    if wiki.is_file():
        sources["wikipedia"] = wiki
    sources = source_files(sources)
    if wiki.is_dir():
        sources["wikipedia"] = {"directory": wiki.name, "members": source_files({
            child.relative_to(wiki).as_posix(): child for child in sorted(wiki.rglob("*"))
            if child.is_file() and not child.name.startswith(".") and child.suffix in (".bz2", ".jsonl")
        })}

    write_json(out / "manifest.json", {
        "dataset": "HotpotQA fullwiki v1 dev / October 2017 Wikipedia introductions",
        "sources": sources, "seed": seed, "available_questions": available, "questions": questions,
        "source_passages": count, "corpus_passages": min(size, count),
        "mapped_support_passages": len(forced), "background_passages": min(size, count) - len(forced),
        "question_sampling": "BLAKE2b-64(seed, domain, ID) and ID",
        "background_sampling": "BLAKE2b-64(seed, domain, ID) and ID" if size < count else "complete",
        "supporting_facts": sum(statuses.values()), "statuses": dict(statuses),
        "mapped_fraction": statuses["mapped"] / sum(statuses.values()),
        "fully_mapped_questions": fully_mapped, "fully_mapped_fraction": fully_mapped / questions,
    })
