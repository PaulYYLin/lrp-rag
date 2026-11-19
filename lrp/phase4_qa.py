"""Build the exact reader inputs and generate answers with a pinned local model."""
import hashlib
import json
import platform
from collections import defaultdict
from itertools import batched
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .io import fingerprint, read_json, read_rows, run_directory, write_json, write_rows

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_FILES = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
BUDGET = 2048
BATCH = 4
SYSTEM = "Answer the question with only the short answer. Use the context when available. Do not explain."
USER = "Context:\n{context}\n\nQuestion: {question}"


def digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def pack(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.input_ids
    count = min(len(ids), BUDGET)
    # A UTF-8 code point can straddle two byte-level BPE tokens.
    while 0 < count < len(ids) and encoded.offset_mapping[count-1][1] > encoded.offset_mapping[count][0]:
        count -= 1
    kept = ids[:count]
    visible = tokenizer.decode(kept, clean_up_tokenization_spaces=False)
    if not text.startswith(visible) or (count == len(ids) and visible != text):
        raise ValueError("Tokenizer offsets do not reproduce the context prefix")
    return kept, visible, len(ids)


def evidence(question, pages, ranking, condition):
    facts = {(f["gold_title"], f["gold_sentence_index"]): f for f in question["supporting_facts"]}
    if not facts:
        raise ValueError("Empty gold evidence")
    starts, ends, blocks = {}, {}, []
    if condition == "gold":
        by_title = defaultdict(list)
        for (title, i), fact in facts.items():
            if fact["gold_sentence"] is not None:
                by_title[title].append((i, fact["gold_sentence"]))
        for title, sentences in by_title.items():
            block = title + "\n"
            base = sum(len(b) + 2 for b in blocks)
            for i, sentence in sorted(sentences):
                block += sentence
                ends[title, i] = base + len(block)
            blocks.append(block)
    else:
        cursor = 0
        for pid in ranking:
            page = pages[pid]
            starts[pid] = cursor + len(page["title"]) + 1
            block = page["title"] + "\n" + page["text"]
            blocks.append(block)
            cursor += len(block) + 2
        for key, fact in facts.items():
            pid = fact.get("page_id")
            if fact["status"] == "mapped" and pid in starts:
                body = pages[pid]["text"]
                if body[fact["char_start"]:fact["char_end"]] != fact["sentence"]:
                    raise ValueError(f"Evidence span disagrees with corpus for {key}")
                ends[key] = starts[pid] + fact["char_end"]
    return "\n\n".join(blocks), facts, ends


def make_input(tokenizer, question, pages, ranking, condition):
    text, facts, ends = evidence(question, pages, ranking, condition)
    context_ids, visible, original_tokens = pack(tokenizer, text)
    # Encode the context separately so the reader receives exactly the scored IDs.
    marker = "{LRP_CONTEXT}"
    user = USER.format(context=marker, question=question["text"])
    template = tokenizer.apply_chat_template([dict(role="system", content=SYSTEM), dict(role="user", content=user)],
                                             tokenize=False, add_generation_prompt=True)
    prefix, suffix = template.split(marker)
    encode = lambda t: tokenizer.encode(t, add_special_tokens=False)
    ids = encode(prefix) + context_ids + encode(suffix)
    statuses = []
    for key, fact in facts.items():
        if key in ends:
            status = "visible" if ends[key] <= len(visible) else "truncated"
        else:
            status = "not_retrieved" if fact["status"] == "mapped" else "unmapped"
        statuses.append([*key, status])
    count = sum(s[2] == "visible" for s in statuses)
    retrieved_titles = {title for title, _, status in statuses if status in ("visible", "truncated")}
    return ids, dict(id=question["id"], passage_ids=ranking, context=visible, context_tokens=len(context_ids),
                     before_truncation_tokens=original_tokens, facts=statuses, prompt_sha256=digest(ids),
                     fully_mapped=all(f["status"] == "mapped" for f in facts.values()),
                     sentence_recall=count/len(facts), complete=int(count == len(facts)),
                     passage_recall=len(retrieved_titles)/len({key[0] for key in facts}))


def generate(args, tokenizer, prompts, settings):
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = args.generation_cache / signature
    model = None
    results, batches = {}, []
    ordered = sorted(prompts, key=lambda key: (len(prompts[key]), key))
    config = GenerationConfig(**settings["decoding"])
    for n, keys in enumerate(batched(ordered, BATCH)):
        key = hashlib.sha256("".join(keys).encode()).hexdigest()
        path = directory / f"{key}.json"
        if path.exists():
            batch = read_json(path)
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.generator, local_files_only=True,
                        dtype=getattr(torch, settings["dtype"]), attn_implementation="sdpa").to(args.device).eval()
            inputs = tokenizer.pad({"input_ids": [prompts[k] for k in keys]}, padding=True,
                                   return_attention_mask=True, return_tensors="pt").to(args.device)
            start = perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=config)
            new_tokens = output[:, inputs.input_ids.shape[1]:].cpu().tolist()
            seconds = perf_counter() - start
            predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            batch = dict(keys=list(keys), predictions=predictions, seconds=seconds)
            write_json(path, batch)
        if batch["keys"] != list(keys):
            raise ValueError("Mismatched answer cache")
        results.update(zip(keys, batch["predictions"], strict=True))
        batches.append(dict(keys=batch["keys"], seconds=batch["seconds"]))
        if n % 25 == 0:
            print(f"Generated {len(results)}/{len(prompts)} distinct prompts", flush=True)
    return results, batches


def answer(args, controls):
    for name, expected in MODEL_FILES.items():
        if fingerprint(args.generator / name) != expected:
            raise ValueError(f"Generator snapshot mismatch: {name}")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.generator, local_files_only=True, use_fast=True,
                                              padding_side="left")
    questions_file = args.prepared / "eval-hotpot.jsonl"
    questions = list(read_rows(questions_file))
    if len(set(q["id"] for q in questions)) != len(questions):
        raise ValueError("Duplicate questions")
    rankings, pages = {}, {}
    if controls:
        conditions = ["question_only", "gold"]
    else:
        meta = read_json(args.retrieval / "manifest.json")
        corpus = args.prepared / "corpus.jsonl"
        if fingerprint(corpus) != meta["corpus_sha256"] or fingerprint(questions_file) != meta["questions_sha256"]:
            raise ValueError("Prepared inputs differ from retrieval")
        conditions = list(meta["points"])
        needed = set()
        for name in conditions:
            path = args.retrieval / f"{name}.jsonl"
            if fingerprint(path) != meta["rankings_sha256"][name]:
                raise ValueError("Ranking has changed")
            ranking = {r["id"]: r["passage_ids"] for r in read_rows(path)}
            if ranking.keys() != {q["id"] for q in questions}:
                raise ValueError("Mismatched ranking IDs")
            rankings[name] = ranking
            needed.update(pid for ids in ranking.values() for pid in ids)
        pages = {p["id"]: p for p in read_rows(corpus) if p["id"] in needed}
    settings = dict(model=MODEL, revision=REVISION, files=MODEL_FILES, system_prompt=SYSTEM,
                    user_prompt=USER, context_budget=BUDGET, batch_size=BATCH, device=args.device,
                    dtype="float32" if args.device == "cpu" else "float16", attention="sdpa",
                    torch=torch.__version__, transformers=transformers.__version__, platform=platform.platform(),
                    decoding=dict(max_new_tokens=32, do_sample=False, num_beams=1, repetition_penalty=1.0,
                                  eos_token_id=[151645, 151643], pad_token_id=151643, use_cache=True))
    with run_directory(args.out) as out:
        records, prompts = {}, {}
        for name in conditions:
            records[name] = []
            for q in questions:
                ranking = [] if controls else rankings[name][q["id"]]
                ids, row = make_input(tokenizer, q, pages, ranking, name)
                records[name].append(row)
                prompts[row["prompt_sha256"]] = ids
        predictions, batches = generate(args, tokenizer, prompts, settings)
        for name, rows in records.items():
            for row in rows:
                row["prediction"] = predictions[row["prompt_sha256"]]
            write_rows(out / f"{name}.jsonl", rows)
        write_rows(out / "generation-batches.jsonl", batches)
        write_json(out / "manifest.json", dict(settings=settings, questions_sha256=fingerprint(questions_file),
                   retrieval_sha256=None if controls else fingerprint(args.retrieval / "manifest.json"),
                   conditions=conditions, distinct_prompts=len(prompts), rows=sum(map(len, records.values())),
                   generation_seconds=sum(b["seconds"] for b in batches),
                   rows_sha256={name: fingerprint(out / f"{name}.jsonl") for name in records}))
