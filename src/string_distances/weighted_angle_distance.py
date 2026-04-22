from collections import Counter
import math
from typing import Callable, Dict, Optional, Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Helpers: n-gram counting + cosine-angle distance on count vectors
# ---------------------------------------------------------------------
def _ngram_counts(s: str, n: int) -> Counter[str]:
    """Return Counter of all contiguous n-grams in s (with multiplicity)."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n > len(s):
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _l2_norm_sq(counts: Counter[str]) -> int:
    """Squared L2 norm of a nonnegative integer vector stored as a Counter."""
    return sum(v * v for v in counts.values())


def _dot(a: Counter[str], b: Counter[str]) -> int:
    """Dot product of two sparse count vectors stored as Counters."""
    # Iterate on smaller support for speed
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0) for k, v in a.items())


def _angle_distance_from_counts(a: Counter[str], b: Counter[str]) -> float:
    na2 = _l2_norm_sq(a)
    nb2 = _l2_norm_sq(b)

    if na2 == 0 and nb2 == 0:
        return 0.0
    if na2 == 0 or nb2 == 0:
        return math.pi / 2

    dot_value = _dot(a, b)

    # Numerically stabler than sqrt(na2) * sqrt(nb2)
    denom = math.sqrt(na2 * nb2)
    cos_value = dot_value / denom

    # Snap extremely-close values to avoid acos(0.9999999999999998) artifacts
    if cos_value >= 1.0 - 1e-15:
        cos_value = 1.0
    elif cos_value <= 0.0 + 1e-15:
        cos_value = 0.0
    else:
        cos_value = max(0.0, min(1.0, cos_value))

    return math.acos(cos_value)


# ---------------------------------------------------------------------
# Your metric: naive weighted-angle distance
# ---------------------------------------------------------------------
def naive_weighted_angle_distance(
    s: str,
    t: str,
    *,
    rho: float = 0.5,
    max_n: Optional[int] = None,
) -> float:
    """
    Naive implementation of d_rho(s,t) = sum_{n>=1} rho^n * theta_n(s,t),
    where theta_n compares the n-gram count vectors by angle distance.

    Parameters
    ----------
    rho:
        Exponential decay parameter. (Your theory allows rho>0, but most
        experiments will use 0<rho<1.)
    max_n:
        Optional cap on n to speed up experiments. If None, uses
        max(len(s), len(t)), which matches the exact definition.

    Notes
    -----
    This is naive: it recomputes n-gram counts for each n. Complexity is
    roughly O(max_n * (len(s)+len(t))) with large constants and memory churn.
    Your suffix-based method will ultimately replace this for long sequences.
    """
    if rho <= 0:
        raise ValueError("rho must be > 0")

    L = max(len(s), len(t))
    # We cap at max_n for simplicity
    N = L if max_n is None else min(max_n, L)

    total = 0.0
    for n in range(1, N + 1):
        cs = _ngram_counts(s, n)
        ct = _ngram_counts(t, n)
        theta_n = _angle_distance_from_counts(cs, ct)
        total += (rho**n) * theta_n
    return total


def kgram_cosine_angle_distance(s: str, t: str, *, k: int) -> float:
    """
    Fixed-k version: theta_k(s,t) only (angle distance between k-gram counts).
    This is one of your baselines.
    """
    return _angle_distance_from_counts(_ngram_counts(s, k), _ngram_counts(t, k))


# ---------------------------------------------------------------------
# Fast path: suffix array + LCP interval processing
# ---------------------------------------------------------------------
def _suffix_array_doubling(u: str) -> list[int]:
    """
    O(n log n) suffix array (doubling). Good enough in Python for moderate n.
    If you want *really* fast, swap this for pydivsufsort/divsufsort.
    """
    n = len(u)
    if n == 0:
        return []

    sa = list(range(n))
    rank = [ord(c) for c in u]
    tmp = [0] * n
    k = 1

    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        tmp[sa[0]] = 0
        for idx in range(1, n):
            i, j = sa[idx - 1], sa[idx]
            prev = (rank[i], rank[i + k] if i + k < n else -1)
            cur = (rank[j], rank[j + k] if j + k < n else -1)
            tmp[j] = tmp[i] + (prev < cur)

        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2

    return sa


def _lcp_kasai(u: str, sa: Sequence[int]) -> list[int]:
    """Kasai LCP: returns LCP[i] = lcp(sa[i], sa[i+1])."""
    n = len(u)
    if n == 0:
        return []
    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i

    h = 0
    lcp = [0] * (n - 1)
    for i in range(n):
        r = rank[i]
        if r == n - 1:
            h = 0
            continue
        j = sa[r + 1]
        while i + h < n and j + h < n and u[i + h] == u[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1
    return lcp


@dataclass(frozen=True)
class _LCPInterval:
    lcp: int
    left_suffix: int  # inclusive SA index
    right_suffix: int  # inclusive SA index
    parent_lcp: int


def _enumerate_lcp_intervals(lcp: Sequence[int]) -> list[_LCPInterval]:
    """
    Enumerate all LCP-intervals (internal nodes of the suffix tree),
    with parent LCP depth, via a standard monotone stack.

    Each interval corresponds to some node v with depth=lcp, spanning
    suffix array indices [left_suffix, right_suffix].
    """
    stack: list[tuple[int, int]] = []  # (lcp_value, left_lcp_index)
    out: list[_LCPInterval] = []

    for i, cur in enumerate(lcp):
        left = i
        while stack and stack[-1][0] > cur:
            h, lft = stack.pop()
            parent_h = max(cur, stack[-1][0] if stack else 0)
            # interval spans LCP indices [lft, i-1], hence SA indices [lft, i]
            out.append(
                _LCPInterval(
                    lcp=h, left_suffix=lft, right_suffix=i, parent_lcp=parent_h
                )
            )
            left = lft
        if not stack or stack[-1][0] < cur:
            stack.append((cur, left))

    # flush
    i = len(lcp)
    while stack:
        h, lft = stack.pop()
        parent_h = stack[-1][0] if stack else 0
        # interval spans SA indices [lft, len(lcp)]
        out.append(
            _LCPInterval(
                lcp=h, left_suffix=lft, right_suffix=len(lcp), parent_lcp=parent_h
            )
        )

    return out


def _gst_aggregated_stats(
    s: str,
    t: str,
    *,
    max_n: Optional[int] = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Compute A_n(S), A_n(T), B_n(S,T) for n=1..N where N=max_n or max(|s|,|t|),
    using U = S#T$ and suffix-array/LCP-interval processing.

    Returns lists sized N+1 with index 0 unused.
    """
    m, n = len(s), len(t)
    N = max(m, n)
    if max_n is not None:
        N = min(N, max_n)
    if N == 0:
        return [0], [0], [0]

    U = s + "#" + t + "$"
    sa = _suffix_array_doubling(U)
    lcp = _lcp_kasai(U, sa)

    # leaf origin flags in SA order
    originS = [0] * len(sa)
    originT = [0] * len(sa)

    for i, pos in enumerate(sa):
        if pos < m:
            originS[i] = 1
        elif m < pos < m + 1 + n:
            originT[i] = 1

    # prefix sums for O(1) interval counts
    PS = [0]
    PT = [0]
    for i in range(len(sa)):
        PS.append(PS[-1] + originS[i])
        PT.append(PT[-1] + originT[i])

    def countS(l: int, r: int) -> int:
        return PS[r + 1] - PS[l]

    def countT(l: int, r: int) -> int:
        return PT[r + 1] - PT[l]

    # clean length to the next sentinel for each suffix start position in U
    # (this is only needed for *leaf edges* to avoid counting substrings that include # or $)
    cleanlim = [0] * len(U)
    hash_pos = m
    dollar_pos = len(U) - 1
    for pos in range(len(U)):
        if pos < hash_pos:
            cleanlim[pos] = hash_pos - pos
        elif pos == hash_pos:
            cleanlim[pos] = 0
        elif pos < dollar_pos:
            cleanlim[pos] = dollar_pos - pos
        else:
            cleanlim[pos] = 0

    # difference arrays for A_S, A_T, B
    DS = [0] * (N + 3)
    DT = [0] * (N + 3)
    DST = [0] * (N + 3)

    # internal nodes (LCP intervals)
    for node in _enumerate_lcp_intervals(lcp):
        if node.lcp <= 0:
            continue
        lo = node.parent_lcp + 1
        hi = min(node.lcp, N)
        if lo > hi:
            continue

        occS = countS(node.left_suffix, node.right_suffix)
        occT = countT(node.left_suffix, node.right_suffix)

        a = occS * occS
        b = occT * occT
        c = occS * occT

        DS[lo] += a
        DS[hi + 1] -= a
        DT[lo] += b
        DT[hi + 1] -= b
        DST[lo] += c
        DST[hi + 1] -= c

    # leaf edges: contribute the “unique tail” beyond the max LCP with neighbors
    for i, pos in enumerate(sa):
        lim = cleanlim[pos]
        if lim <= 0:
            continue

        lprev = lcp[i - 1] if i > 0 else 0
        lnext = lcp[i] if i < len(lcp) else 0
        parent_depth = max(lprev, lnext)

        lo = parent_depth + 1
        hi = min(lim, N)
        if lo > hi:
            continue

        os = 1 if pos < m else 0
        ot = 1 if (m < pos < m + 1 + n) else 0

        DS[lo] += os
        DS[hi + 1] -= os
        DT[lo] += ot
        DT[hi + 1] -= ot
        DST[lo] += os * ot
        DST[hi + 1] -= os * ot

    # prefix sums to get A_n, B_n
    A_S = [0] * (N + 1)
    A_T = [0] * (N + 1)
    B = [0] * (N + 1)

    acc = 0
    for k in range(1, N + 1):
        acc += DS[k]
        A_S[k] = acc

    acc = 0
    for k in range(1, N + 1):
        acc += DT[k]
        A_T[k] = acc

    acc = 0
    for k in range(1, N + 1):
        acc += DST[k]
        B[k] = acc

    return A_S, A_T, B


def weighted_angle_distance(
    s: str,
    t: str,
    *,
    rho: float = 0.5,
    max_n: Optional[int] = None,
    use_suffix_array: bool = True,
) -> float:
    """
    Fast weighted-angle distance using suffix-array GST stats (default).
    Falls back to naive if requested.

    This computes:
      d_rho(s,t) = sum_{n>=1} rho^n * theta_n(s,t),
    where theta_n is the angle distance between n-gram count vectors.
    """
    if rho <= 0:
        raise ValueError("rho must be > 0")

    if not use_suffix_array:
        return naive_weighted_angle_distance(s, t, rho=rho, max_n=max_n)

    # handle empties cheaply
    if not s and not t:
        return 0.0
    if not s:
        L = len(t)
        N = L if max_n is None else min(max_n, L)
        return (math.pi / 2) * sum(rho**k for k in range(1, N + 1))
    if not t:
        L = len(s)
        N = L if max_n is None else min(max_n, L)
        return (math.pi / 2) * sum(rho**k for k in range(1, N + 1))

    A_S, A_T, B = _gst_aggregated_stats(s, t, max_n=max_n)
    N = len(A_S) - 1

    total = 0.0
    for n in range(1, N + 1):
        a = A_S[n]
        b = A_T[n]
        c = B[n]

        if a == 0 and b == 0:
            theta = 0.0
        elif a == 0 or b == 0:
            theta = math.pi / 2
        else:
            cosv = c / math.sqrt(a * b)
            cosv = max(0.0, min(1.0, cosv))
            theta = math.acos(cosv)

        total += (rho**n) * theta

    return total
