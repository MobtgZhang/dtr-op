"""Checkpointed, parallel experiment runner with rate limiting.

Every task result is appended to a JSONL file as soon as it finishes, so runs
can be interrupted and resumed.  Each record stores the full provenance:
task, gold, prediction, metrics, retrieved cases, prompts and raw LLM output.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..llm_client import LLMClient
from .metrics import aggregate

_write_lock = threading.Lock()
_request_lock = threading.Lock()
_last_request_time = 0
MIN_REQUEST_INTERVAL = 2.0  # Minimum seconds between API requests


def _load_done(out_file: Path) -> Dict[str, Dict]:
    done = {}
    if out_file.is_file():
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["task_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _append(out_file: Path, rec: Dict) -> None:
    with _write_lock:
        with open(out_file, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _rate_limit():
    """Ensure minimum interval between API requests."""
    global _last_request_time
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()


def run_agent_over(
    agent,
    test_items: List[Dict],
    evaluate_fn: Callable[[List[str], List[str]], Dict],
    out_file: Path,
    label: str,
    max_workers: int = 1,  # Default to sequential to avoid rate limits
    progress_every: int = 20,
) -> Dict:
    """Run `agent.query(task)` over test_items with rate limiting; returns aggregate metrics."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_file)
    todo = [t for t in test_items if t["task_id"] not in done]
    print(f"[{label}] {len(done)} cached, {len(todo)} to run", flush=True)

    results = list(done.values())
    counter = {"n": len(done)}

    def work(item: Dict) -> Dict:
        task = item["task"]
        gold = item["gold_actions"]
        t0 = time.time()
        rec: Dict = {
            "task_id": item["task_id"],
            "task": task,
            "gold": gold,
            "meta": {k: v for k, v in item.items() if k not in ("task", "gold_actions")},
        }
        try:
            # Rate limit before making API calls
            _rate_limit()
            
            out = agent.query(task)
            pred = out.get("pred", [])
            if not isinstance(pred, list):
                pred = []
            rec.update({
                "pred": pred,
                "retrieved_cases": out.get("retrieved_cases", []),
                "prompt": out.get("prompt", ""),
                "raw_response": out.get("raw_response", ""),
                "extra": {k: v for k, v in out.items()
                          if k not in ("pred", "retrieved_cases", "prompt", "raw_response")},
            })
        except Exception as exc:
            rec.update({"pred": [], "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed"] = round(time.time() - t0, 2)
        # WebShop需要instruction，传递完整rec
        if hasattr(evaluate_fn, '__name__') and 'webshop' in evaluate_fn.__name__:
            rec["metrics"] = evaluate_fn(rec["pred"], gold, task=task)
        else:
            rec["metrics"] = evaluate_fn(rec["pred"], gold)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(work, item): item for item in todo}
            for fut in as_completed(futures):
                rec = fut.result()
                results.append(rec)
                _append(out_file, rec)
                counter["n"] += 1
                if counter["n"] % progress_every == 0 or counter["n"] == len(test_items):
                    agg = aggregate([r["metrics"] for r in results])
                    print(f"[{label}] {counter['n']}/{len(test_items)} "
                          f"SR={agg['sr']:.1f} GC={agg['gc']:.1f} PLWGC={agg['plw_gc']:.1f}", flush=True)

    agg = aggregate([r["metrics"] for r in results])
    agg["label"] = label
    return agg
