import math

import pytest
from hypothesis import given, strategies as st, settings

import src.string_distances.weighted_angle_distance as wad


ALPHABET = "abcd"  # avoid '#' and '$' sentinels


@settings(max_examples=200, deadline=None)
@given(
    s=st.text(ALPHABET, min_size=0, max_size=8),
    t=st.text(ALPHABET, min_size=0, max_size=8),
    rho=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    max_n=st.one_of(st.none(), st.integers(min_value=0, max_value=8)),
)
def test_fast_matches_naive_property(s, t, rho, max_n):
    fast = wad.weighted_angle_distance(s, t, rho=rho, max_n=max_n, use_suffix_array=True)
    naive = wad.naive_weighted_angle_distance(s, t, rho=rho, max_n=max_n)
    assert fast == pytest.approx(naive, rel=1e-10, abs=1e-10)


@settings(max_examples=200, deadline=None)
@given(
    s=st.text(ALPHABET, min_size=0, max_size=10),
    t=st.text(ALPHABET, min_size=0, max_size=10),
    rho=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
)
def test_symmetry(s, t, rho):
    d1 = wad.weighted_angle_distance(s, t, rho=rho, use_suffix_array=True)
    d2 = wad.weighted_angle_distance(t, s, rho=rho, use_suffix_array=True)
    assert d1 == pytest.approx(d2, rel=1e-12, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    s=st.text(ALPHABET, min_size=0, max_size=10),
    rho=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    max_n=st.one_of(st.none(), st.integers(min_value=0, max_value=10)),
)
def test_identity_and_nonnegativity(s, rho, max_n):
    d = wad.weighted_angle_distance(s, s, rho=rho, max_n=max_n, use_suffix_array=True)
    assert d >= -1e-12  # numerical wiggle
    assert d == pytest.approx(0.0, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    s=st.text(ALPHABET, min_size=0, max_size=10),
    t=st.text(ALPHABET, min_size=0, max_size=10),
    rho=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    max_n=st.one_of(st.none(), st.integers(min_value=0, max_value=10)),
)
def test_upper_bound_by_pi_over_2(s, t, rho, max_n):
    d = wad.weighted_angle_distance(s, t, rho=rho, max_n=max_n, use_suffix_array=True)
    L = max(len(s), len(t))
    N = L if max_n is None else min(max_n, L)
    upper = (math.pi / 2) * sum(rho**k for k in range(1, N + 1))
    assert 0.0 <= d <= upper + 1e-9


@settings(max_examples=200, deadline=None)
@given(
    s=st.text(ALPHABET, min_size=0, max_size=10),
    t=st.text(ALPHABET, min_size=0, max_size=10),
    rho=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    k1=st.integers(min_value=0, max_value=10),
    k2=st.integers(min_value=0, max_value=10),
)
def test_monotone_in_max_n(s, t, rho, k1, k2):
    lo = min(k1, k2)
    hi = max(k1, k2)

    d_lo = wad.weighted_angle_distance(s, t, rho=rho, max_n=lo, use_suffix_array=True)
    d_hi = wad.weighted_angle_distance(s, t, rho=rho, max_n=hi, use_suffix_array=True)

    assert d_lo <= d_hi + 1e-12
