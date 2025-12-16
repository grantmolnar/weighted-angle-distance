from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

DistanceFn = Callable[[str, str], float]


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
    """
    Angle distance theta(u,v) in [0, pi/2] for nonnegative vectors u,v
    represented as Counters (sparse counts), with the same conventions as
    your paper:
      - theta(0,0) = 0
      - theta(0,v) = pi/2 if exactly one is zero
      - otherwise arccos( <u,v> / (||u|| ||v||) )
    """
    na2 = _l2_norm_sq(a)
    nb2 = _l2_norm_sq(b)

    if na2 == 0 and nb2 == 0:
        return 0.0
    if na2 == 0 or nb2 == 0:
        return math.pi / 2

    dot = _dot(a, b)
    denom = math.sqrt(na2) * math.sqrt(nb2)
    cos = dot / denom

    # Numerical safety (should already be in [0,1] for nonneg vectors)
    cos = max(0.0, min(1.0, cos))
    return math.acos(cos)


# ---------------------------------------------------------------------
# Your metric: naive weighted-angle distance
# ---------------------------------------------------------------------
def weighted_angle_distance(
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
# Baselines: Levenshtein (fast if RapidFuzz installed; fallback otherwise)
# ---------------------------------------------------------------------
def _levenshtein_fallback(a: str, b: str) -> int:
    """
    Pure-Python Levenshtein distance (unit costs), O(len(a)*len(b)).
    Fine for small strings; use RapidFuzz for large-scale runs.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter string to reduce memory if desired
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def levenshtein_distance(a: str, b: str) -> float:
    """
    Levenshtein distance wrapper:
    - uses RapidFuzz if available
    - otherwise uses a pure-python DP fallback
    """
    try:
        from rapidfuzz.distance.Levenshtein import distance as rf_lev  # type: ignore
        return float(rf_lev(a, b))
    except Exception:
        return float(_levenshtein_fallback(a, b))


# ---------------------------------------------------------------------
# Optional extras (only enabled if deps exist)
# ---------------------------------------------------------------------
def jaro_winkler_distance(a: str, b: str) -> float:
    """
    Jaro-Winkler *distance* in [0,1] if RapidFuzz is available.
    If unavailable, raises ImportError (so you notice and can add the dep).
    """
    try:
        from rapidfuzz.distance.JaroWinkler import distance as rf_jw  # type: ignore
    except Exception as e:
        raise ImportError(
            "jaro_winkler_distance requires rapidfuzz. Install with `pip install rapidfuzz`."
        ) from e
    return float(rf_jw(a, b))


# ---------------------------------------------------------------------
# Registry: one place to grab all distances you want to test
# ---------------------------------------------------------------------
def get_distance_registry(
    *,
    rho_values: Sequence[float] = (0.5, (1+5**0.5)/2),
    k_values: Sequence[int] = (2, 3, 4),
    max_n_for_weighted: Optional[int] = None,
    include_optional: bool = True,
) -> Dict[str, DistanceFn]:
    """
    Return a dict name -> distance_fn(s,t). Great for looping experiments.

    include_optional:
        If True, attempts to add distances that require optional deps
        (currently Jaro-Winkler via RapidFuzz).
    """
    registry: Dict[str, DistanceFn] = {}

    # Your metric
    for rho in rho_values:
        registry[f"weighted_angle_rho={rho}"] = lambda s, t: weighted_angle_distance(
            s, t, rho=rho, max_n=max_n_for_weighted
        )

    # Fixed-k cosine-angle baselines
    for k in k_values:
        registry[f"kgram_angle_k={k}"] = lambda s, t, k=k: kgram_cosine_angle_distance(s, t, k=k)

    # Edit-distance baseline
    registry["levenshtein"] = levenshtein_distance

    # Optional extras
    if include_optional:
        try:
            _ = jaro_winkler_distance("a", "a")
            registry["jaro_winkler"] = jaro_winkler_distance
        except ImportError:
            # Silently skip: you can choose to be strict instead.
            pass

    return registry
