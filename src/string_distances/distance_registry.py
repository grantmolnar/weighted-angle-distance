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

def damerau_levenshtein_distance(a: str, b: str) -> float:
    """
    Damerau–Levenshtein edit distance (adjacent transpositions count as 1).
    Requires RapidFuzz.
    """
    from rapidfuzz.distance.DamerauLevenshtein import distance as rf_dist  # type: ignore

    return float(rf_dist(a, b))

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


def jensen_shannon_kgram_distance(a: str, b: str, *, k: int = 4, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon *distance* (sqrt(JS divergence)) between k-gram distributions.
    - symmetric
    - bounded (0..~sqrt(log 2) depending on log base)
    - metric when using sqrt(JS)

    Implementation uses NumPy only.
    """
    ca = _kgram_counts(a, k)
    cb = _kgram_counts(b, k)

    if not ca and not cb:
        return 0.0
    if not ca or not cb:
        # maximally different in the absence of overlap; keep simple + stable
        # (you can tune this; DBSCAN doesn’t require strict metricity)
        return 1.0

    vocab = list(set(ca.keys()) | set(cb.keys()))
    pa = np.array([ca.get(g, 0.0) for g in vocab], dtype=np.float64)
    pb = np.array([cb.get(g, 0.0) for g in vocab], dtype=np.float64)

    pa = pa / pa.sum()
    pb = pb / pb.sum()

    # smooth to avoid log(0); keep normalization
    pa = np.clip(pa, eps, 1.0)
    pb = np.clip(pb, eps, 1.0)
    pa = pa / pa.sum()
    pb = pb / pb.sum()

    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    js_div = 0.5 * (kl(pa, m) + kl(pb, m))
    return float(np.sqrt(max(js_div, 0.0)))

# ---------------------------------------------------------------------
# Registry: one place to grab all distances you want to test
# ---------------------------------------------------------------------
def get_distance_registry(
    *,
    rho_values: Sequence[float] = (0.5, (1 + 5**0.5) / 2),
    k_values: Sequence[int] = (2, 3, 4),
    max_n_for_weighted: Optional[int] = None,
    include_optional: bool = True,
    js_k_values: Sequence[int] = (3, 4),          # NEW
    ncd_compressors: Sequence[str] = ("lzma",),   # NEW: can add "zlib"
) -> Dict[str, DistanceFn]:
    registry: Dict[str, DistanceFn] = {}

    def make_weighted_angle(rho: float) -> DistanceFn:
        def dist(s: str, t: str) -> float:
            return weighted_angle_distance(s, t, rho=rho, max_n=max_n_for_weighted)
        return dist

    def make_kgram_angle(k: int) -> DistanceFn:
        def dist(s: str, t: str) -> float:
            return kgram_cosine_angle_distance(s, t, k=k)
        return dist

    for rho in rho_values:
        registry[f"weighted_angle_rho={rho}"] = make_weighted_angle(float(rho))

    for k in k_values:
        registry[f"kgram_angle_k={k}"] = make_kgram_angle(int(k))

    registry["levenshtein"] = levenshtein_distance

    if include_optional:
        # LCS via RapidFuzz
        try:
            _ = longest_common_subsequence_length("a", "a")
            registry["lcs"] = longest_common_subsequence_length
        except ImportError:
            pass

        # (1) Damerau–Levenshtein via RapidFuzz
        try:
            from src.string_distances.rapidfuzz_extras import damerau_levenshtein_distance
            _ = damerau_levenshtein_distance("ab", "ba")
            registry["damerau_levenshtein"] = damerau_levenshtein_distance
        except Exception:
            pass

        # (2) Jensen–Shannon distance on k-gram distributions (NumPy)
        try:
            from src.string_distances.js_kgram_distance import jensen_shannon_kgram_distance

            for k in js_k_values:
                kk = int(k)
                registry[f"js_kgram_k={kk}"] = (lambda kk=kk: (lambda s, t: jensen_shannon_kgram_distance(s, t, k=kk)))()
        except Exception:
            pass

    return registry