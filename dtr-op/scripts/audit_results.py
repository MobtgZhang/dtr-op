"""Read and deduplicate the historical generation archive."""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Sequence


SR_THRESHOLD = {"alfworld": 0.70, "scienceworld": 0.20}


def load_jsonl(path: Path) -> List[dict]:
    """Read and validate every JSON line from an archived run."""
    with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dedupe_by_task(records: Sequence[dict]) -> List[dict]:
    """Select one record per task_id, preferring a *successful* generation.

    Selection rule (in priority order, then the later archive line wins):

      1. a non-error record with a response or prediction;
      2. a non-error record without a response or prediction;
      3. an error record.

    Rule 1 over rule 3 is the important one: without it a transient network
    error silently overwrites a correct generation and is then counted as a
    wrong answer, which biases the baseline downwards.
    """
    best: Dict[str, dict] = {}
    rank: Dict[str, int] = {}
    for r in records:
        key = r.get("task_id", str(len(best)))
        if r.get("error"):
            cur_rank = 0
        elif (r.get("raw_response") or "").strip() or (r.get("pred") or []):
            cur_rank = 2
        else:
            cur_rank = 1
        if key not in best or cur_rank >= rank[key]:
            best[key] = r
            rank[key] = cur_rank
    return list(best.values())
