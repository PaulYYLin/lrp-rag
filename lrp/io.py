import gzip
import hashlib
import json
import os
import tarfile
import tempfile
import unicodedata
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_file(path, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(mode, dir=path.parent, delete=False, **options) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, data):
    with atomic_file(path) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path, rows):
    with atomic_file(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files(paths):
    return {key: {"file": path.name, "sha256": fingerprint(path)}
            for key, path in paths.items()}


def normalized_query(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sample_key(seed, domain, identifier):
    material = f"{seed}\0{domain}\0{identifier}".encode("utf-8")
    return hashlib.blake2b(material, digest_size=8).digest(), identifier


@contextmanager
def text_source(path, member=None):
    if member and path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r|gz") as archive:
            for entry in archive:
                if entry.isfile() and Path(entry.name).name == member:
                    with archive.extractfile(entry) as stream:
                        yield (line.decode("utf-8") for line in stream)
                    return
        raise ValueError(f"{member} is missing from {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield stream


def id_text(stream):
    for number, line in enumerate(stream, 1):
        fields = line.rstrip("\r\n").split("\t", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].strip():
            raise ValueError(f"Invalid ID/text row {number}")
        yield tuple(fields)


def input_path(directory, *names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing input in {directory}: {', '.join(names)}")


@contextmanager
def run_directory(path):
    """Publish a complete result directory, with the manifest written by its caller."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Choose a new output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}-") as directory:
        working = Path(directory)
        yield working
        os.rename(working, path)
