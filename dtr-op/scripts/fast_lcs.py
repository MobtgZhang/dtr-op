"""Fast exact longest-common-subsequence, verified against the DP reference.

The bit-parallel formulation (Crochemore et al.) computes the LCS length of
two sequences over a small alphabet in O(n*ceil(m/w)) word operations, which
is the difference between minutes and milliseconds on the ScienceWorld
trajectories, whose reference plans can be several hundred steps long.

``lcs_length`` is the exact dynamic-programming reference used to verify
this implementation; ``lcs_length_fast`` is the one used in the analyses.
"""

from __future__ import annotations

from typing import Dict, Sequence


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Exact LCS length, O(len(a) * len(b)) time, O(min) space (reference)."""
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


def lcs_length_fast(a: Sequence[str], b: Sequence[str]) -> int:
    """Exact LCS length, bit-parallel (Crochemore et al., 2001)."""
    if not a or not b:
        return 0
    m = len(b)
    masks: Dict[str, int] = {}
    for j, ch in enumerate(b):
        masks[ch] = masks.get(ch, 0) | (1 << j)
    full = (1 << m) - 1
    v = full
    for ch in a:
        p = masks.get(ch, 0)
        if not p:
            continue
        u = v & p
        v = ((v + u) | (v - u)) & full
    # the number of zero bits of v equals the LCS length
    return m - bin(v).count("1")
