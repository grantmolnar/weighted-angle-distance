import math
from collections import Counter

import pytest

import src.string_distances.weighted_angle_distance as wad


def brute_stats(s: str, t: str, max_n: int | None = None):
    """Brute A_n(S), A_n(T), B_n(S,T) using n-gram Counters (slow but exact for small strings)."""
    L = max(len(s), len(t))
    N = L if max_n is None else min(max_n, L)
    A_S = [0] * (N + 1)
    A_T = [0] * (N + 1)
    B = [0] * (N + 1)
    for n in range(1, N + 1):
        cs = wad._ngram_counts(s, n)
        ct = wad._ngram_counts(t, n)
        A_S[n] = sum(v * v for v in cs.values())
        A_T[n] = sum(v * v for v in ct.values())
        B[n] = sum(v * ct.get(k, 0) for k, v in cs.items())
    return A_S, A_T, B


def test_ngram_counts_basic_and_edges():
    assert wad._ngram_counts("banana", 2) == Counter({"an": 2, "na": 2, "ba": 1})
    assert wad._ngram_counts("abc", 3) == Counter({"abc": 1})
    assert wad._ngram_counts("abc", 4) == Counter()

    with pytest.raises(ValueError):
        wad._ngram_counts("abc", 0)
    with pytest.raises(ValueError):
        wad._ngram_counts("abc", -1)


def test_l2_norm_sq_and_dot_swap_branch():
    counts = Counter({"a": 2, "b": 3})
    assert wad._l2_norm_sq(counts) == 2 * 2 + 3 * 3

    # Force the "swap" branch: len(a) > len(b)
    a = Counter({"x": 1, "y": 2})
    b = Counter({"y": 3})
    assert wad._dot(a, b) == 6  # 2*3


def test_angle_distance_branches():
    # both zero
    assert wad._angle_distance_from_counts(Counter(), Counter()) == 0.0

    # one zero
    assert wad._angle_distance_from_counts(Counter(), Counter({"a": 1})) == pytest.approx(math.pi / 2)

    # identical nonzero -> 0
    c = Counter({"a": 2, "b": 1})
    assert wad._angle_distance_from_counts(c, c) == pytest.approx(0.0)

    # disjoint support -> pi/2
    assert wad._angle_distance_from_counts(Counter({"a": 1}), Counter({"b": 1})) == pytest.approx(math.pi / 2)


def test_naive_weighted_angle_distance_edges():
    with pytest.raises(ValueError):
        wad.naive_weighted_angle_distance("a", "b", rho=0.0)

    assert wad.naive_weighted_angle_distance("", "") == 0.0

    rho = 0.3
    # s empty => each theta_n = pi/2 for n<=|t|
    expected = (math.pi / 2) * sum(rho**k for k in range(1, len("abcd") + 1))
    assert wad.naive_weighted_angle_distance("", "abcd", rho=rho) == pytest.approx(expected)


def test_kgram_cosine_angle_distance_basic():
    # k too large => both k-gram counters empty => theta=0
    assert wad.kgram_cosine_angle_distance("ab", "xyz", k=10) == 0.0

    # one empty (k>len(s)) and other non-empty (k<=len(t)) => pi/2
    assert wad.kgram_cosine_angle_distance("a", "abcd", k=2) == pytest.approx(math.pi / 2)

    # identical strings => 0 for any valid k
    assert wad.kgram_cosine_angle_distance("banana", "banana", k=2) == pytest.approx(0.0)


def test_suffix_array_doubling_correctness_and_empty():
    assert wad._suffix_array_doubling("") == []

    u = "banana"
    sa = wad._suffix_array_doubling(u)
    # Compare to brute sorted suffixes
    expected = sorted(range(len(u)), key=lambda i: u[i:])
    assert sa == expected


def test_lcp_kasai_correctness_and_empty():
    assert wad._lcp_kasai("", []) == []

    u = "banana"
    sa = wad._suffix_array_doubling(u)
    lcp = wad._lcp_kasai(u, sa)

    def brute_lcp(i: int, j: int) -> int:
        k = 0
        while i + k < len(u) and j + k < len(u) and u[i + k] == u[j + k]:
            k += 1
        return k

    expected = [brute_lcp(sa[i], sa[i + 1]) for i in range(len(sa) - 1)]
    assert lcp == expected


def test_enumerate_lcp_intervals_exercises_stack_paths():
    # crafted to force push, pop, and flush behaviors
    lcp = [1, 3, 3, 0, 2, 2, 1, 0]
    intervals = wad._enumerate_lcp_intervals(lcp)

    # sanity: should produce some intervals, including one with lcp=3 and one with lcp=2
    lcps = [iv.lcp for iv in intervals]
    assert 3 in lcps
    assert 2 in lcps
    assert len(intervals) > 0

    # ensure fields are coherent
    for iv in intervals:
        assert iv.left_suffix <= iv.right_suffix
        assert iv.parent_lcp >= 0


def test_gst_aggregated_stats_matches_bruteforce_small_strings():
    s, t = "ababa", "baba"
    A1, A2, B = wad._gst_aggregated_stats(s, t, max_n=None)
    A1b, A2b, Bb = brute_stats(s, t, max_n=None)
    assert A1 == A1b
    assert A2 == A2b
    assert B == Bb

    # also check with a max_n cap
    A1, A2, B = wad._gst_aggregated_stats(s, t, max_n=2)
    A1b, A2b, Bb = brute_stats(s, t, max_n=2)
    assert A1 == A1b
    assert A2 == A2b
    assert B == Bb


def test_gst_aggregated_stats_N_zero_path():
    A1, A2, B = wad._gst_aggregated_stats("a", "b", max_n=0)
    assert A1 == [0]
    assert A2 == [0]
    assert B == [0]


def test_weighted_angle_distance_fast_equals_naive_on_examples():
    for s, t, rho, max_n in [
        ("", "", 0.5, None),
        ("", "abcd", 0.7, None),
        ("abcd", "", 0.7, 3),
        ("banana", "bananas", 0.4, None),
        ("ababa", "baba", 0.8, 3),
    ]:
        fast = wad.weighted_angle_distance(s, t, rho=rho, max_n=max_n, use_suffix_array=True)
        naive = wad.naive_weighted_angle_distance(s, t, rho=rho, max_n=max_n)
        assert fast == pytest.approx(naive, rel=1e-10, abs=1e-10)


def test_weighted_angle_distance_fallback_path_equals_naive():
    s, t = "ababa", "baba"
    fast_off = wad.weighted_angle_distance(s, t, rho=0.6, max_n=3, use_suffix_array=False)
    naive = wad.naive_weighted_angle_distance(s, t, rho=0.6, max_n=3)
    assert fast_off == pytest.approx(naive, rel=1e-12, abs=1e-12)


def test_weighted_angle_distance_rho_validation():
    with pytest.raises(ValueError):
        wad.weighted_angle_distance("a", "b", rho=0.0)


def test_weighted_angle_distance_hits_all_theta_branches_via_monkeypatch(monkeypatch):
    # This forces the internal loop to execute all three theta branches:
    #  (a==0 and b==0), (a==0 or b==0), and (else)
    def fake_stats(s: str, t: str, *, max_n=None):
        # N=3
        # n=1: a=0,b=0 -> theta=0
        # n=2: a=0,b=9 -> theta=pi/2
        # n=3: a=4,b=9,c=6 -> else-branch, cos=1 -> theta=0
        return [0, 0, 0, 4], [0, 0, 9, 9], [0, 0, 0, 6]

    monkeypatch.setattr(wad, "_gst_aggregated_stats", fake_stats)

    out = wad.weighted_angle_distance("x", "y", rho=0.5, max_n=None, use_suffix_array=True)
    # expected = rho^1*0 + rho^2*(pi/2) + rho^3*0
    assert out == pytest.approx((0.5**2) * (math.pi / 2))
