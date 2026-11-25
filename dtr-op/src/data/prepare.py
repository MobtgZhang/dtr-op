"""Read the fixed processed splits shipped with the artifact."""

import gzip
import json

from ..config import DATA_DIR


def prepare(dataset):
    splits = {}
    for split in ("train", "test"):
        path = DATA_DIR / "processed" / f"{dataset}_{split}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            splits[split] = [json.loads(line) for line in stream if line.strip()]
    return splits


def prepare_alfworld():
    return prepare("alfworld")


def prepare_scienceworld():
    return prepare("scienceworld")
