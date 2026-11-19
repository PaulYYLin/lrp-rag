import hashlib
import json
import platform
import sqlite3
from contextlib import closing
from itertools import batched

import numpy as np

from .io import fingerprint

MODEL = "facebook/contriever"
REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"
DIMENSION = 768
CACHE_BLOCK = 256


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("Cannot L2-normalize a zero or non-finite embedding")
    return vectors / norms


class EmbeddingCache:
    def __init__(self, directory, device, batch_size, max_length, threads, model_dir=None):
        import torch
        import transformers

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch.set_num_threads(threads)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.torch = torch
        self.settings = {
            "model": MODEL, "revision": REVISION, "dimension": DIMENSION,
            "max_length": max_length, "pooling": "attention_mask_mean",
            "normalization": "applied after cache lookup", "dtype": "float32", "tokenizer": "fast",
            "device": device, "machine": platform.machine(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "encoding_batch_size": batch_size, "encoding_threads": threads,
        }
        if model_dir is not None:
            self.settings["local_files"] = {
                file.name: fingerprint(file) for file in sorted(model_dir.iterdir())
                if file.is_file() and file.suffix in (".json", ".txt", ".bin", ".safetensors")
            }
        digest = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{digest}.sqlite3"

    def encode(self, texts):
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer

            options = {"local_files_only": True} if self.model_dir else {"revision": REVISION}
            model_name = self.model_dir or MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **options)
            self.model = AutoModel.from_pretrained(model_name, weights_only=True, **options)
            self.model.to(device=self.device, dtype=self.torch.float32).eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            tokens = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = tokens.masked_fill(mask == 0, 0).sum(1) / mask.sum(1)
        vectors = pooled.float().cpu().numpy()
        if vectors.shape != (len(texts), DIMENSION) or not np.all(np.isfinite(vectors)):
            raise ValueError("Contriever returned invalid pooled embeddings")
        return vectors

    def vectors(self, rows, name):
        matrix = np.empty((len(rows), DIMENSION), dtype=np.float32)
        hits = 0
        with closing(sqlite3.connect(self.path, timeout=60)) as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("CREATE TABLE IF NOT EXISTS vectors "
                             "(digest BLOB PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID")
            for start in range(0, len(rows), CACHE_BLOCK):
                block = rows[start:start + CACHE_BLOCK]
                keys = [hashlib.sha256(row["text"].encode()).digest() for row in block]
                placeholders = ",".join("?" for _ in keys)
                cached = dict(database.execute(
                    f"SELECT digest, vector FROM vectors WHERE digest IN ({placeholders})", keys
                ))
                hits += sum(key in cached for key in keys)
                missing = {key: row["text"] for key, row in zip(keys, block) if key not in cached}
                by_length = sorted(missing, key=lambda key: len(missing[key]))
                encoded = {}
                for batch in batched(by_length, self.batch_size):
                    vectors = self.encode([missing[key] for key in batch])
                    encoded.update((key, vector.astype("<f4", copy=False).tobytes())
                                   for key, vector in zip(batch, vectors, strict=True))
                if encoded:
                    with database:
                        changes = database.total_changes
                        database.executemany("INSERT OR IGNORE INTO vectors VALUES (?, ?)", encoded.items())
                        if database.total_changes - changes != len(encoded):
                            # A concurrent run won a write. Use the persisted value in both runs.
                            cached.update(database.execute(
                                f"SELECT digest, vector FROM vectors WHERE digest IN ({placeholders})", keys
                            ))
                        else:
                            cached.update(encoded)
                for index, key in enumerate(keys, start):
                    matrix[index] = np.frombuffer(cached[key], dtype="<f4")
                if encoded and (start + CACHE_BLOCK) % 2048 == 0:
                    print(f"{name}: {min(start + CACHE_BLOCK, len(rows)):,}/{len(rows):,} embeddings", flush=True)
        print(f"{name}: {len(rows):,} embeddings ({hits:,} cached)", flush=True)
        return matrix
