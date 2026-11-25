"""Replay the historical plan parsing and canonicalization."""

from typing import List, Optional, Sequence

from src.canonicalize import canonicalize_actions
from src.prompts import parse_numbered


def context_receps(cases: Sequence[dict]) -> List[str]:
    return [a.split("go to ", 1)[1]
            for c in cases for a in (c.get("actions") or [])
            if a.startswith("go to ")]


def replay(rec: dict, dataset: str) -> Optional[dict]:
    """Reconstruct every pipeline stage from the stored artefacts."""
    raw = rec.get("raw_response") or ""
    refine = (rec.get("extra") or {}).get("refine_response") or ""
    cases = rec.get("retrieved_cases") or []

    plan0 = parse_numbered(raw)
    plan1 = parse_numbered(refine) if refine else []
    used_refine = bool(refine) and len(plan1) > 0
    plan_pre_canon = plan1 if used_refine else plan0

    if dataset == "alfworld":
        pred = canonicalize_actions(plan_pre_canon, context_receps=context_receps(cases))
        pred_nocanon = plan_pre_canon
    else:
        pred = plan_pre_canon
        pred_nocanon = plan_pre_canon

    return {
        "plan0": plan0,
        "plan1": plan1,
        "used_refine": used_refine,
        "pred_replay": pred,
        "pred_nocanon": pred_nocanon,
    }
