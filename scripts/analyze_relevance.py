#!/usr/bin/env python3
"""
analyze_relevance.py
====================
An offline study of the *relevance predicate* of the trajectory database.

Why this script exists
-----------------------
The paper argues that the relevance predicate of a trajectory database should
be judged on the *operator skeleton* of a candidate solution rather than on
the surface text of its description.  This script turns that argument into a
measurement that uses **no model calls at all** and **no test label as an
input to the predicate**.

Fidelity to the deployed system
-------------------------------
The hybrid predicate is not re-implemented here: it is obtained by calling the
shipped ``CaseLibrary.recall`` directly, so the study measures exactly the
operator that produced the numbers in the main table.  The two comparison
predicates are defined on the *same* repository and return the *same number of
cases* as the deployed system at the corresponding budget, so every row of the
comparison is budget-matched.

Budget semantics
----------------
``top_k`` is the shipped agent's own parameter: the number of *topically
ranked* cases.  The deployed operator prepends the family's prototype, so the
retrieved set has ``top_k + 1`` members.  All three predicates are evaluated
at that same set size.

The predicates
--------------
``lexical``        pure content-word overlap, no family term, no prototype.
                   This is what a conventional text-retrieval index returns.
``hybrid``         the shipped operator: content-word overlap plus a task-family
                   bonus, with the family's prototype trajectory prepended.
``structural``     a *leak-free* skeleton-aware predicate: candidates are ranked
                   by the longest-common-subsequence agreement between their
                   operator skeleton and the skeleton of the query's family
                   prototype, plus the same family bonus.
``family_lexical`` a *decomposition* predicate: lexical ranking restricted to
                   the query's own task family, without the prototype.  It is
                   not a candidate method; it exists to attribute the
                   behaviour of the deployed operator to its two parts -- the
                   family restriction and the prototype.

Reported quantities
-------------------
For each predicate and each budget we report, over the test queries:

  * ``coverage`` -- best-case structural coverage: the largest fraction of the
    reference operator skeleton recovered by any single retrieved case.
  * ``precision``-- the fraction of retrieved cases that are themselves
    consistent with the reference skeleton (LCS >= 0.8).
  * ``purity``   -- the fraction of retrieved cases from the query's own
    task family.
  * ``disagree`` -- mean pairwise skeleton disagreement inside the set.

We additionally compute the **corpus ceiling**: for each query, the best
structural coverage attainable by *any* case in the repository.  The residual
(queries for which no case in the corpus matches) bounds what any relevance
predicate can achieve and is a property of the data, not of the ranking.

Nothing here calls a language model; every number is computed from the
training trajectories and the reference plans that are already on disk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from analyze_structure import action_type  # noqa: E402
from fast_lcs import lcs_length_fast  # noqa: E402

# The budgets that the paper reports.  ``top_k`` is the shipped parameter;
# every predicate returns ``top_k + 1`` cases (the hybrid operator spends one
# slot on the family prototype).
BUDGETS = (1, 3, 5, 7, 12)

# A retrieved case counts as "consistent" with the reference when its skeleton
# agrees with the reference skeleton on at least this fraction.
CONSISTENT = 0.8

# The family-match bonus used by src/case_library.py::CaseLibrary.recall.
TYPE_MATCH_BONUS = 5.0


# ---------------------------------------------------------------------------
# Repository construction -- identical to the experiment code
# ---------------------------------------------------------------------------
def build_library(dataset: str, data):
    """Build the shipped case library and return it with the test split."""
    from src.case_library import CaseLibrary
    lib = CaseLibrary(dataset)
    lib.build_from_episodes(data["train"])
    return lib, data["test"]


def query_family(lib, query: str, dataset: str) -> str:
    """The family the shipped operator assigns to a query."""
    if dataset == "alfworld":
        from src.case_library import infer_alf_task_type
        return infer_alf_task_type(query)
    return lib._match_sw_family(query)


# ---------------------------------------------------------------------------
# Operator skeletons (cached)
# ---------------------------------------------------------------------------
_SKEL_CACHE: Dict[str, List[str]] = {}


def skeleton_of(case: dict) -> List[str]:
    """The operator skeleton of a case, cached by case id."""
    cid = case.get("id", "")
    sk = _SKEL_CACHE.get(cid)
    if sk is None:
        sk = [action_type(a) for a in case.get("actions") or []]
        _SKEL_CACHE[cid] = sk
    return sk


def lcs_cov(a: Sequence[str], b: Sequence[str]) -> float:
    """LCS(a, b) normalised by len(b)."""
    if not a or not b:
        return 0.0
    return lcs_length_fast(a, b) / len(b)


# ---------------------------------------------------------------------------
# The comparison predicates
# ---------------------------------------------------------------------------
def rank_lexical(lib, query: str, n: int) -> List[dict]:
    """Pure content-word overlap; no family term, no prototype.

    This is what a conventional text-retrieval index returns.
    """
    from src.case_library import keywords
    qkw = keywords(query)
    scored = [(float(len(qkw & c["_kw"])), c) for c in lib.cases]
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [c for _, c in scored[:n]]


def rank_structural(lib, proto: Optional[dict], family: str, query: str,
                    n: int) -> List[dict]:
    """Skeleton-aware ranking against the family prototype (leak-free).

    The prototype is mined from the *training* split, so this predicate never
    sees a test plan and it never sees a test entity.
    """
    if proto is None:
        return []
    target = skeleton_of(proto)
    scored = []
    for c in lib.cases:
        score = lcs_cov(skeleton_of(c), target)
        if c["group"] and c["group"] == family:
            score += TYPE_MATCH_BONUS
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [c for _, c in scored[:n]]


def rank_family(lib, query: str, family: str, n: int) -> List[dict]:
    """Family-restricted lexical ranking, without the prototype.

    This isolates the effect of the deployed operator's family bonus: the
    ranking is still lexical, but candidates are drawn only from the query's
    own task family.  Comparing it against ``lexical`` tells us whether the
    family restriction helps or hurts structural coverage.
    """
    from src.case_library import keywords
    qkw = keywords(query)
    scored = [(float(len(qkw & c["_kw"])), c) for c in lib.cases
              if c["group"] == family]
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [c for _, c in scored[:n]]


# ---------------------------------------------------------------------------
# Scoring of a retrieved set
# ---------------------------------------------------------------------------
def score_set(picked: Sequence[dict], gold_actions: Sequence[str],
              family: str) -> dict:
    if not picked or not gold_actions:
        return {"coverage": 0.0, "precision": 0.0, "purity": 0.0,
                "disagreement": 0.0, "n_conflicting": 0.0}
    gskel = [action_type(a) for a in gold_actions]
    covs = [lcs_cov(skeleton_of(c), gskel) for c in picked]
    arr = np.array(covs)
    consistent = arr >= CONSISTENT
    purity = sum(1 for c in picked if c["group"] == family) / len(picked)
    skels = [skeleton_of(c) for c in picked]
    ds = []
    for i in range(len(skels)):
        for j in range(i + 1, len(skels)):
            a, b = skels[i], skels[j]
            if not a or not b:
                continue
            ds.append(1.0 - lcs_length_fast(a, b) / max(len(a), len(b)))
    return {
        "coverage": float(arr.max()),
        "precision": float(consistent.mean()),
        "purity": float(purity),
        "disagreement": float(np.mean(ds)) if ds else 0.0,
        "n_conflicting": float(len(picked) - consistent.sum()),
    }


# ---------------------------------------------------------------------------
# The corpus ceiling: the best coverage attainable by *any* stored case
# ---------------------------------------------------------------------------
def corpus_ceiling(lib, test: Sequence[dict]) -> dict:
    """How much of each reference plan exists somewhere in the repository?

    This is computed over the whole repository, not over a retrieved set, so
    it is independent of the relevance predicate.  The residual -- queries for
    which no stored case matches -- is the part of the problem that no
    ranking policy can solve.
    """
    covs = []
    for t in test:
        gold = t.get("gold_actions") or []
        if not gold:
            continue
        gskel = [action_type(a) for a in gold]
        best = 0.0
        for c in lib.cases:
            v = lcs_cov(skeleton_of(c), gskel)
            if v > best:
                best = v
                if best >= 1.0:
                    break
        covs.append(best)
    arr = np.array(covs)
    return {
        "n": int(arr.size),
        "mean_best_coverage_pct": round(100 * float(arr.mean()), 1),
        "frac_ge_80_pct": round(100 * float((arr >= 0.8).mean()), 1),
        "frac_ge_50_pct": round(100 * float((arr >= 0.5).mean()), 1),
        "frac_no_match_pct": round(100 * float((arr < 0.8).mean()), 1),
    }


# ---------------------------------------------------------------------------
def analyse(dataset: str, lib, test: Sequence[dict]) -> dict:
    print(f"=== {dataset}: {len(lib.cases)} cases, {len(test)} test queries ===")

    # Pre-compute the family of every test query once.
    fams = {t.get("task_id", str(i)): query_family(lib, t.get("task", ""), dataset)
            for i, t in enumerate(test)}
    protos = {g: lib.prototypes.get(g) for g in set(fams.values())}

    out: Dict[str, dict] = {}
    for name in ("lexical", "hybrid", "structural", "family_lexical"):
        out[name] = {}
        for K in BUDGETS:
            n = K + 1  # the deployed operator returns top_k + 1 cases
            rows = []
            t0 = time.perf_counter()
            for t in test:
                q = t.get("task", "")
                gold = t.get("gold_actions") or []
                fam = fams.get(t.get("task_id", ""), "")
                if name == "hybrid":
                    # The shipped operator itself -- no re-implementation.
                    picked = lib.recall(q, top_k=K, use_prototype=True)
                elif name == "lexical":
                    picked = rank_lexical(lib, q, n)
                elif name == "family_lexical":
                    picked = rank_family(lib, q, fam, n)
                else:
                    picked = rank_structural(lib, protos.get(fam), fam, q, n)
                rows.append(score_set(picked, gold, fam))
            dt = (time.perf_counter() - t0) / max(len(test), 1)
            cov = np.array([r["coverage"] for r in rows])
            out[name][str(K)] = {
                "n": len(rows),
                "n_cases": n,
                "coverage_pct": round(100 * float(cov.mean()), 1),
                "frac_ge_80_pct": round(100 * float((cov >= 0.8).mean()), 1),
                "precision_pct": round(100 * float(np.mean([r["precision"] for r in rows])), 1),
                "purity_pct": round(100 * float(np.mean([r["purity"] for r in rows])), 1),
                "disagreement_pct": round(100 * float(np.mean([r["disagreement"] for r in rows])), 1),
                "n_conflicting": round(float(np.mean([r["n_conflicting"] for r in rows])), 2),
                "ms_per_query": round(1000 * dt, 3),
            }
            v = out[name][str(K)]
            print(f"  {name:<11} K={K:<3} cover={v['coverage_pct']:>5.1f}%  "
                  f">=80%:{v['frac_ge_80_pct']:>5.1f}%  prec={v['precision_pct']:>5.1f}%  "
                  f"purity={v['purity_pct']:>5.1f}%  disagree={v['disagreement_pct']:>5.1f}%  "
                  f"conflicting={v['n_conflicting']:>4.2f}")
    return out


def family_heterogeneity(lib) -> dict:
    """How heterogeneous is each task family?

    A family is heterogeneous when its members disagree a lot with each other,
    so that the family prototype is a poor stand-in for an individual member.
    """
    by_family: Dict[str, List[dict]] = {}
    for c in lib.cases:
        if c["group"]:
            by_family.setdefault(c["group"], []).append(c)
    out = {}
    for fam, grp in by_family.items():
        if len(grp) < 2:
            continue
        sk = [skeleton_of(c) for c in grp[:60]]
        ds = []
        for i in range(len(sk)):
            for j in range(i + 1, len(sk)):
                a, b = sk[i], sk[j]
                if not a or not b:
                    continue
                ds.append(1.0 - lcs_length_fast(a, b) / max(len(a), len(b)))
        out[fam] = {
            "n_cases": len(grp),
            "mean_disagreement_pct": round(100 * float(np.mean(ds)), 1) if ds else None,
        }
    vals = [v["mean_disagreement_pct"] for v in out.values()
            if v["mean_disagreement_pct"] is not None]
    return {
        "per_family": out,
        "mean_disagreement_pct": round(float(np.mean(vals)), 1) if vals else None,
        "n_families": len(out),
    }


