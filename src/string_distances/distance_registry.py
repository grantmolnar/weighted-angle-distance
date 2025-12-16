from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence
from src.string_distances.weighted_angle_distance import (
    weighted_angle_distance,
    kgram_cosine_angle_distance,
)

DistanceFn = Callable[[str, str], float]


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


def longest_common_subsequence_length(a: str, b: str) -> float:
    """
    Levenshtein distance wrapper:
    - uses RapidFuzz if available
    - otherwise uses a pure-python DP fallback
    """
    try:
        from rapidfuzz.distance.LCSseq import distance as rf_lcs  # type: ignore

        return float(rf_lcs(a, b))
    except Exception:
        raise ImportError


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
    rho_values: Sequence[float] = (0.5, (1 + 5**0.5) / 2),
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
        registry[f"weighted_angle_rho={rho}"] = (
            lambda s, t, rho=rho: weighted_angle_distance(
                s, t, rho=rho, max_n=max_n_for_weighted
            )
        )

    # Fixed-k cosine-angle baselines
    for k in k_values:
        registry[f"kgram_angle_k={k}"] = lambda s, t, k=k: kgram_cosine_angle_distance(
            s, t, k=k
        )

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
            # Optional extras
        try:
            _ = longest_common_subsequence_length("a", "a")
            registry["lcs"] = longest_common_subsequence_length
        except ImportError:
            pass

    return registry
