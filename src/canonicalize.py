"""Action-grammar canonicalization for ALFWorld.

ALFWorld's TextWorld grammar (see alfworld.agents.utils.misc templates) uses:

    go to <recep> | open/close <recep> | take <obj> from <recep>
    move <obj> to <recep> | use <lamp> | heat/cool/clean/slice <obj> with <recep>
    examine <thing> | inventory | look

LLMs frequently drift from this grammar ("put X in Y", "place X on Y",
"pick up X", bare "cool X").  Canonicalization maps predictions back onto the
benchmark grammar before evaluation.  ScienceWorld uses free-form commands, so
it is not canonicalized.
"""
import re
from typing import List, Optional

_PUT_RE = re.compile(r"^(?:put|place|set|drop|position)\s+(.+?)\s+(?:in|on|into|onto|in/on)\s+(.+)$", re.I)
_PICKUP_RE = re.compile(r"^(?:pick up|pickup|grab|take)\s+(.+?)\s+from\s+(.+)$", re.I)
_COOL_RE = re.compile(r"^cool\s+(.+)$", re.I)
_HEAT_RE = re.compile(r"^heat\s+(.+)$", re.I)
_CLEAN_RE = re.compile(r"^clean\s+(.+)$", re.I)


def canonicalize_action(action: str, context_receps: Optional[List[str]] = None) -> str:
    a = action.strip().strip('"').strip("'").strip(".")
    if not a:
        return a
    a = re.sub(r"\s+", " ", a)

    m = _PUT_RE.match(a)
    if m:
        obj, target = m.group(1).strip(), m.group(2).strip()
        return f"move {obj} to {target}"

    m = _PICKUP_RE.match(a)
    if m:
        obj, src = m.group(1).strip(), m.group(2).strip()
        return f"take {obj} from {src}"

    def _with_appliance(verb: str, obj: str, hint: str) -> str:
        if context_receps:
            for r in context_receps:
                if hint in r.lower():
                    return f"{verb} {obj} with {r}"
        return f"{verb} {obj} with {hint} 1"

    for regex, verb, hint in ((_COOL_RE, "cool", "fridge"), (_HEAT_RE, "heat", "microwave"), (_CLEAN_RE, "clean", "sinkbasin")):
        m = regex.match(a)
        if m and " with " not in a.lower():
            return _with_appliance(verb, m.group(1).strip(), hint)

    return a.lower()


def canonicalize_actions(actions: List[str], context_receps: Optional[List[str]] = None) -> List[str]:
    out = []
    for a in actions:
        if not a or not a.strip():
            continue
        c = canonicalize_action(a, context_receps)
        if c:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Normalization used by the *evaluator* (applied to both pred and gold)
# ---------------------------------------------------------------------------
def normalize_action(action: str) -> str:
    """Normalize an ALFWorld command for comparison (pred AND gold)."""
    a = action.strip().lower()
    a = re.sub(r"\s+", " ", a).strip(".")
    m = _PUT_RE.match(a)
    if m:
        return f"move {m.group(1).strip()} to {m.group(2).strip()}"
    return a
