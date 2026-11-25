"""Rescore archived predictions and compute episode bootstrap intervals."""

from typing import List, Sequence

import numpy as np

from analyze_replay import replay
from audit_results import SR_THRESHOLD
from src.evaluation.metrics import evaluate_alf_offline, evaluate_sw_offline


SEED = 20271111


def score(dataset: str, recs: Sequence[dict], disable_refinement: bool = True
          ) -> List[dict]:
    out = []
    for r in recs:
        rp = replay(r, dataset)
        if rp is None:
            continue
        gold = r.get("gold") or []
        if disable_refinement:
            plan = rp["plan0"]
            if dataset == "alfworld":
                from src.canonicalize import canonicalize_actions
                receps = [a.split("go to ", 1)[1]
                          for c in (r.get("retrieved_cases") or [])
                          for a in (c.get("actions") or [])
                          if a.startswith("go to ")]
                plan = canonicalize_actions(plan, context_receps=receps)
        else:
            plan = rp["pred_replay"]
        fn = evaluate_alf_offline if dataset == "alfworld" else evaluate_sw_offline
        m = fn(list(plan), list(gold))
        out.append({
            "task_id": r.get("task_id"),
            "gc": float(m["gc"]),
            "plw_gc": float(m["plw_gc"]),
            "n_pred": len(plan),
            "pred": list(plan),
            "gold": list(gold),
        })
    return out


def summarize(dataset: str, recs: Sequence[dict]) -> dict:
    if not recs:
        return {"n": 0}
    tau = SR_THRESHOLD[dataset]
    g = np.array([r["gc"] for r in recs]) * 100
    p = np.array([r["plw_gc"] for r in recs]) * 100
    # Bootstrap CI of the *mean*, not a percentile of the raw values.
    rng = np.random.default_rng(SEED)
    bi = rng.integers(0, len(g), size=(10000, len(g)))
    gboot = g[bi].mean(axis=1)
    pboot = p[bi].mean(axis=1)
    return {
        "n": len(recs),
        "sr_at_th": round(100 * float((g >= 100 * tau).mean()), 1),
        "gc": round(float(g.mean()), 2),
        "gc_ci": [round(float(np.percentile(gboot, 2.5)), 2),
                  round(float(np.percentile(gboot, 97.5)), 2)],
        "plwgc": round(float(p.mean()), 2),
        "plwgc_ci": [round(float(np.percentile(pboot, 2.5)), 2),
                     round(float(np.percentile(pboot, 97.5)), 2)],
        "gc_median": round(float(np.median(g)), 2),
        "gc_std": round(float(np.std(g)), 2),
    }


def paired(a: List[dict], b: List[dict], metric: str = "gc") -> dict:
    ia = {r["task_id"]: r[metric] for r in a}
    ib = {r["task_id"]: r[metric] for r in b}
    common = sorted(set(ia) & set(ib))
    if not common:
        return {"n_paired": 0}
    d = (np.array([ia[t] for t in common])
         - np.array([ib[t] for t in common])) * 100
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "n_paired": len(common),
        "mean_diff": round(float(d.mean()), 2),
        "ci_low": round(float(np.percentile(boot, 2.5)), 2),
        "ci_high": round(float(np.percentile(boot, 97.5)), 2),
        "p_a_ge_b": round(float((boot >= 0).mean()), 4),
    }


def _score_stored(dataset: str, recs: Sequence[dict]) -> List[dict]:
    """Score a stored run using its own recorded predictions."""
    out = []
    evaluate = evaluate_alf_offline if dataset == "alfworld" else evaluate_sw_offline
    for r in recs:
        pred, gold = r["pred"], r["gold"]
        metrics = evaluate(pred, gold)
        assert all(metrics[key] == r["metrics"][key] for key in ("gc", "plw_gc")), r["task_id"]
        out.append({
            "task_id": r.get("task_id"),
            "gc": float(metrics["gc"]),
            "plw_gc": float(metrics["plw_gc"]),
            "n_pred": len(pred),
            "pred": pred,
            "gold": gold,
        })
    return out
