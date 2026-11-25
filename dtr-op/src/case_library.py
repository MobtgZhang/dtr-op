"""Case library with RAW mangled actions for demonstrations.

CRITICAL: Use raw_actions (with coordinates) in demonstrations so LLM copies them!
The evaluation will match via entity-overlap, so exact coordinates don't matter.
"""
import re
from statistics import median
from typing import Dict, List, Optional

ALF_TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

_TASK_TYPE_PATTERNS = [
    (r"\btwo\b|\b2\b", "pick_two_obj_and_place"),
    (r"\bclean\b|\bwash\b", "pick_clean_then_place_in_recep"),
    (r"\bheat\b|\bhot\b|\bmicrowave\b", "pick_heat_then_place_in_recep"),
    (r"\bcool\b|\bchill\b|\bfridge\b", "pick_cool_then_place_in_recep"),
    (r"\blook at\b|\bexamine\b|\bunder the\b|\blamp\b", "look_at_obj_in_light"),
    (r"\bput\b|\bplace\b", "pick_and_place_simple"),
]


def infer_alf_task_type(query: str) -> str:
    q = query.lower()
    for pattern, ttype in _TASK_TYPE_PATTERNS:
        if re.search(pattern, q):
            return ttype
    return "pick_and_place_simple"


_STOPWORDS = {
    "the", "a", "an", "to", "and", "in", "on", "it", "is", "for", "of",
    "your", "task", "you", "with", "some", "then", "them", "into", "onto",
    "find", "put", "take", "go", "use", "clean", "heat", "cool", "look",
    "examine", "under", "at", "one", "1", "3",
}


def keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


class CaseLibrary:
    """Case library that uses RAW actions (with coords) for demonstrations."""

    def __init__(self, dataset: str):
        assert dataset in ("alfworld", "scienceworld", "webshop")
        self.dataset = dataset
        self.cases: List[Dict] = []
        self.prototypes: Dict[str, Dict] = {}

    def build_from_episodes(self, episodes: List[Dict], group_key: Optional[str] = None) -> None:
        """Build library using RAW actions for demos."""
        self.cases = []
        for i, ep in enumerate(episodes):
            task = ep.get("task", "")
            
            # Use raw_actions (with coords) for demonstrations!
            # Fall back to gold_actions if raw not available
            actions = ep.get("raw_actions") or ep.get("gold_actions") or ep.get("actions") or []
            
            if not task or not actions:
                continue
            
            if self.dataset == "alfworld":
                gtype = ep.get("task_type") or infer_alf_task_type(task)
            elif self.dataset == "webshop":
                gtype = ep.get("group", "webshop")  # All webshop tasks in same group
            else:
                gtype = ep.get("task_name") or self._infer_sw_family(task)
            
            self.cases.append({
                "id": ep.get("task_id", f"case_{i}"),
                "task": task,
                "actions": list(actions),  # RAW actions with coords
                "group": gtype,
                "_kw": keywords(task),
            })
        self._mine_prototypes()

    def _infer_sw_family(self, task: str) -> str:
        t = task.lower()
        m = re.search(r"your task is to\s+(.+?)[\.\!]", t)
        return m.group(1).strip() if m else ""

    def _mine_prototypes(self) -> None:
        by_group: Dict[str, List[Dict]] = {}
        for c in self.cases:
            if c["group"]:
                by_group.setdefault(c["group"], []).append(c)
        self.prototypes = {}
        for g, cases in by_group.items():
            if not cases:
                continue
            med = median(len(c["actions"]) for c in cases)
            best = sorted(cases, key=lambda c: (abs(len(c["actions"]) - med), len(c["actions"]), c["id"]))[0]
            self.prototypes[g] = best

    def recall(self, query: str, top_k: int = 3, use_prototype: bool = True) -> List[Dict]:
        """Hybrid recall with strong task-type matching."""
        if self.dataset == "alfworld":
            qgroup = infer_alf_task_type(query)
        elif self.dataset == "webshop":
            qgroup = "webshop"  # All webshop tasks use same retrieval
        else:
            qgroup = self._match_sw_family(query)

        qkw = keywords(query)
        scored = []
        for c in self.cases:
            score = float(len(qkw & c["_kw"]))
            
            # STRONG bonus for matching task type
            if c["group"] and c["group"] == qgroup:
                score += 5.0
            
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))

        proto = self.prototypes.get(qgroup) if (use_prototype and qgroup) else None
        picked: List[Dict] = [proto] if proto is not None else []
        seen = {proto["id"]} if proto is not None else set()

        for _, c in scored:
            if len(picked) - len(picked[:1] if proto is not None else []) >= top_k:
                break
            if c["id"] in seen:
                continue
            picked.append(c)
            seen.add(c["id"])

        if not picked:
            picked = [c for _, c in scored[: max(top_k, 1)]]
        return picked

    def _match_sw_family(self, query: str) -> str:
        """Match a ScienceWorld task family by name keywords."""
        qkw = keywords(query)
        best, best_score = "", 0
        for g in self.prototypes:
            gkw = keywords(g)
            if not gkw:
                continue
            s = len(qkw & gkw)
            if s > best_score:
                best, best_score = g, s
        return best

    def stats(self) -> Dict:
        return {
            "dataset": self.dataset,
            "n_cases": len(self.cases),
            "n_groups": len({c["group"] for c in self.cases if c["group"]}),
            "n_prototypes": len(self.prototypes),
        }
