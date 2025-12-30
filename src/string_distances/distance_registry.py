from __future__ import annotations

from collections import Counter
from functools import lru_cache
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
    """Pure-Python Levenshtein distance (unit costs), O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter string to reduce memory.
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


@lru_cache(maxsize=1)
def _rf_levenshtein_distance():
    """Cached RapidFuzz Levenshtein distance callable (or None if unavailable)."""
    try:
        from rapidfuzz.distance.Levenshtein import distance as rf_lev  # type: ignore

        return rf_lev
    except Exception:
        return None


@lru_cache(maxsize=1)
def _rf_process_cdist():
    """Cached rapidfuzz.process.cdist (or None if unavailable)."""
    try:
        from rapidfuzz.process import cdist  # type: ignore

        return cdist
    except Exception:
        return None


def levenshtein_distance(a: str, b: str) -> float:
    """
    Levenshtein distance wrapper.
    Uses RapidFuzz if available, otherwise falls back to pure Python.
    """
    rf_lev = _rf_levenshtein_distance()
    if rf_lev is not None:
        return float(rf_lev(a, b))
    return float(_levenshtein_fallback(a, b))


def _levenshtein_pairwise(sequences: Sequence[str]):
    """
    Pairwise Levenshtein distance matrix.

    If RapidFuzz is installed, uses rapidfuzz.process.cdist (fast).
    Otherwise falls back to an O(N^2) Python loop.
    """
    import numpy as np

    rf_lev = _rf_levenshtein_distance()
    cdist = _rf_process_cdist()
    if rf_lev is not None and cdist is not None:
        D = cdist(sequences, sequences, scorer=rf_lev, dtype=np.float64)
        return np.asarray(D, dtype=float)

    n = len(sequences)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = sequences[i]
        for j in range(i + 1, n):
            d = levenshtein_distance(si, sequences[j])
            D[i, j] = d
            D[j, i] = d
    return D


# Allow callers (e.g. DBSCAN) to detect a fast path without changing DistanceFn.
setattr(levenshtein_distance, "pairwise", _levenshtein_pairwise)


def damerau_levenshtein_distance(a: str, b: str) -> float:
    """Damerau–Levenshtein distance (falls back to Levenshtein if RapidFuzz missing)."""
    try:
        from rapidfuzz.distance.DamerauLevenshtein import distance as rf_dam  # type: ignore

        return float(rf_dam(a, b))
    except ImportError:
        # Fallback has no transposition operation, but keeps things working
        return float(_levenshtein_fallback(a, b))


def longest_common_subsequence_length(a: str, b: str) -> float:
    """LCSseq distance wrapper (RapidFuzz required)."""
    try:
        from rapidfuzz.distance import LCSseq  # type: ignore

        return float(LCSseq.distance(a, b))
    except ImportError as e:
        raise ImportError("RapidFuzz not installed; LCSseq distance unavailable") from e


def _kgram_counts(s: str, k: int) -> Counter[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(s) < k:
        return Counter()
    return Counter(s[i : i + k] for i in range(len(s) - k + 1))


def jensen_shannon_kgram_distance(a: str, b: str, k: int = 3, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon *distance* between k-gram distributions (base-2 log).
    Returns a value in [0, 1] for typical inputs.
    """
    import numpy as np

    ca = _kgram_counts(a, k)
    cb = _kgram_counts(b, k)

    # No k-grams at all -> treat as identical.
    if not ca and not cb:
        return 0.0
    # Only one side has mass -> maximal separation for our use-case.
    if not ca or not cb:
        return 1.0

    keys = sorted(set(ca) | set(cb))
    pa = np.array([ca.get(key, 0) for key in keys], dtype=float)
    pb = np.array([cb.get(key, 0) for key in keys], dtype=float)

    pa = pa / (pa.sum() + eps)
    pb = pb / (pb.sum() + eps)
    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    js_div = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return float(np.sqrt(max(js_div, 0.0)))


def get_distance_registry(
    *,
    rho_values: Sequence[float] = (0.5, (1 + 5**0.5) / 2),
    k_values: Sequence[int] = (2, 3, 4),
    max_n_for_weighted: Optional[int] = None,
    include_optional: bool = True,
    js_k_values: Sequence[int] = (3, 4),
    ncd_compressors: Sequence[str] = ("lzma",),
) -> Dict[str, DistanceFn]:
    """
    Build a registry of distance functions, keyed by a short name.

    Notes:
    - Optional distances are included only when include_optional=True.
    - js_kgram distances are implemented in this module (no extra imports).
    - ncd_compressors is reserved for future/optional NCD support.
    """
    registry: Dict[str, DistanceFn] = {}

    def make_weighted_angle(rho: float) -> DistanceFn:
        def dist(s: str, t: str) -> float:
            return weighted_angle_distance(s, t, rho=rho, max_n=max_n_for_weighted)

        return dist

    def make_kgram_angle(k: int) -> DistanceFn:
        def dist(s: str, t: str) -> float:
            return kgram_cosine_angle_distance(s, t, k=k)

        return dist

    # Baselines
    registry["levenshtein"] = levenshtein_distance

    # Weighted-angle family
    for rho in rho_values:
        registry[f"weighted_angle_rho={rho}"] = make_weighted_angle(rho)

    # k-gram angle family
    for k in k_values:
        registry[f"kgram_angle_k={k}"] = make_kgram_angle(k)

    if include_optional:
        # LCS via RapidFuzz
        try:
            _ = longest_common_subsequence_length("a", "a")
            registry["lcs"] = longest_common_subsequence_length
        except ImportError:
            pass

        registry["damerau_levenshtein"] = damerau_levenshtein_distance

        # Jensen-Shannon k-gram distances
        for k in js_k_values:
            kk = int(k)
            registry[f"js_kgram_k={kk}"] = (
                lambda kk=kk: (lambda s, t: jensen_shannon_kgram_distance(s, t, k=kk))
            )()

    # ncd_compressors intentionally unused for now, to keep the registry pure-python
    _ = ncd_compressors

    return registry
