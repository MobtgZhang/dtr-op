"""Extract operator skeletons and score ordered operator agreement."""

import re
from typing import Sequence


ACTION_TYPES = {
    "go", "take", "move", "open", "close", "use", "heat", "cool", "clean",
    "examine", "slice", "put", "place", "pick", "inventory", "look",
    "activate", "deactivate", "focus", "wait", "pour", "mix", "connect",
}


def action_type(action: str) -> str:
    words = re.findall(r"[a-z]+", (action or "").lower())
    for w in words:
        if w in ACTION_TYPES:
            return w
    return words[0] if words else ""


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two operator sequences."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[-1]


def ordered_operator_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Order-sensitive F1 over the operator sequences.

    This is the strict counterpart of the shipped evaluator's order-blind
    entity-overlap score.  It asks: does the predicted plan visit the same
    operators in the same order?
    """
    p = [action_type(a) for a in pred]
    g = [action_type(a) for a in gold]
    if not p or not g:
        return 0.0
    l = lcs_length(p, g)
    if l == 0:
        return 0.0
    prec = l / len(p)
    rec = l / len(g)
    return 2 * prec * rec / (prec + rec)
