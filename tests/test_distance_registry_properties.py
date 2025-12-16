from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import src.string_distances.distance_registry as dr


def _force_rapidfuzz_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = ModuleType("rapidfuzz")
    monkeypatch.setitem(sys.modules, "rapidfuzz", stub)
    for k in list(sys.modules.keys()):
        if k.startswith("rapidfuzz."):
            monkeypatch.delitem(sys.modules, k, raising=False)


@settings(max_examples=120)
@given(st.text(min_size=0, max_size=12), st.text(min_size=0, max_size=12))
def test_levenshtein_fallback_properties(a: str, b: str) -> None:
    d_ab = dr._levenshtein_fallback(a, b)
    d_ba = dr._levenshtein_fallback(b, a)

    assert isinstance(d_ab, int)
    assert d_ab >= 0
    assert d_ab == d_ba
    assert dr._levenshtein_fallback(a, a) == 0

    # upper bound: <= max(len(a),len(b)) with unit costs
    assert d_ab <= max(len(a), len(b))


@settings(max_examples=80)
@given(
    st.text(min_size=0, max_size=10),
    st.text(min_size=0, max_size=10),
    st.text(min_size=0, max_size=10),
)
def test_levenshtein_triangle_inequality(a: str, b: str, c: str) -> None:
    dab = dr._levenshtein_fallback(a, b)
    dbc = dr._levenshtein_fallback(b, c)
    dac = dr._levenshtein_fallback(a, c)
    assert dac <= dab + dbc


@settings(max_examples=80)
@given(st.text(min_size=0, max_size=20), st.text(min_size=0, max_size=20))
def test_registry_distances_are_finite_nonnegative_symmetric_without_optionals(
    a: str, b: str
) -> None:
    reg = dr.get_distance_registry(
        rho_values=(0.5, 1.618),
        k_values=(2, 3),
        include_optional=False,
    )

    for name, fn in reg.items():
        d1 = float(fn(a, b))
        d2 = float(fn(b, a))
        assert np.isfinite(d1)
        assert np.isfinite(d2)
        assert d1 >= 0.0
        assert d2 >= 0.0
        assert d1 == pytest.approx(d2)


def test_registry_with_include_optional_true_is_safe_when_rapidfuzz_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This should not explode; it should simply skip optionals.
    _force_rapidfuzz_unavailable(monkeypatch)
    reg = dr.get_distance_registry(include_optional=True)
    assert "jaro_winkler" not in reg
    # Intended behavior:
    assert "lcs" not in reg
