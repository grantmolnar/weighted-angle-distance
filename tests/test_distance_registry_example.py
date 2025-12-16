from __future__ import annotations

import sys
from types import ModuleType
import numpy as np
import pytest

import src.string_distances.distance_registry as dr
from src.string_distances.weighted_angle_distance import (
    naive_weighted_angle_distance,
    kgram_cosine_angle_distance,
)


def _install_fake_rapidfuzz(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lev_value: float = 123.0,
    jw_value: float = 0.42,
    lcs_value: float = 7.0,
) -> None:
    """
    Install a fake 'rapidfuzz' package into sys.modules so that imports like:
      from rapidfuzz.distance.Levenshtein import distance
    succeed deterministically, without requiring rapidfuzz to be installed.
    """
    rf = ModuleType("rapidfuzz")
    rf.__path__ = []  # mark as package
    dist = ModuleType("rapidfuzz.distance")
    dist.__path__ = []  # mark as package

    lev = ModuleType("rapidfuzz.distance.Levenshtein")
    lev.distance = lambda a, b: lev_value  # type: ignore[attr-defined]

    jw = ModuleType("rapidfuzz.distance.JaroWinkler")
    jw.distance = lambda a, b: jw_value  # type: ignore[attr-defined]

    lcsseq = ModuleType("rapidfuzz.distance.LCSseq")
    lcsseq.distance = lambda a, b: lcs_value  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rapidfuzz", rf)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", dist)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.Levenshtein", lev)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.JaroWinkler", jw)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.LCSseq", lcsseq)


def _force_rapidfuzz_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Force submodule imports (rapidfuzz.distance.*) to fail, even if rapidfuzz is
    installed, by inserting a non-package 'rapidfuzz' stub.
    """
    stub = ModuleType("rapidfuzz")  # no __path__ => not a package
    monkeypatch.setitem(sys.modules, "rapidfuzz", stub)
    # remove any already-loaded submodules
    for k in list(sys.modules.keys()):
        if k.startswith("rapidfuzz."):
            monkeypatch.delitem(sys.modules, k, raising=False)


def test__levenshtein_fallback_basic_examples() -> None:
    assert dr._levenshtein_fallback("", "") == 0
    assert dr._levenshtein_fallback("", "abc") == 3
    assert dr._levenshtein_fallback("abc", "") == 3
    assert dr._levenshtein_fallback("kitten", "sitting") == 3
    assert dr._levenshtein_fallback("abc", "abc") == 0


def test_levenshtein_distance_uses_fallback_when_rapidfuzz_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    assert dr.levenshtein_distance("kitten", "sitting") == 3.0


def test_levenshtein_distance_uses_rapidfuzz_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rapidfuzz(monkeypatch, lev_value=999.0)
    assert dr.levenshtein_distance("anything", "else") == 999.0


def test_jaro_winkler_distance_raises_importerror_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    with pytest.raises(ImportError) as e:
        dr.jaro_winkler_distance("a", "a")
    assert "rapidfuzz" in str(e.value).lower()


def test_jaro_winkler_distance_works_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rapidfuzz(monkeypatch, jw_value=0.77)
    assert dr.jaro_winkler_distance("a", "b") == 0.77


def test_lcs_raises_importerror_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    with pytest.raises(ImportError):
        dr.longest_common_subsequence_length("abc", "abc")


def test_lcs_works_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rapidfuzz(monkeypatch, lcs_value=11.0)
    assert dr.longest_common_subsequence_length("abc", "zzz") == 11.0


def test_get_distance_registry_core_keys_present_and_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep optionals off for determinism
    reg = dr.get_distance_registry(rho_values=(0.5,), k_values=(2, 3), include_optional=False)

    assert "levenshtein" in reg
    assert "kgram_angle_k=2" in reg
    assert "kgram_angle_k=3" in reg
    assert "weighted_angle_rho=0.5" in reg

    for k, fn in reg.items():
        out = fn("abc", "abd")
        assert isinstance(out, float)
        assert np.isfinite(out)


def test_get_distance_registry_kgram_binds_k_correctly() -> None:
    reg = dr.get_distance_registry(rho_values=(0.5,), k_values=(2, 4), include_optional=False)
    s, t = "GATTACA", "GACTATA"

    for k in (2, 4):
        key = f"kgram_angle_k={k}"
        assert key in reg
        got = reg[key](s, t)
        exp = float(kgram_cosine_angle_distance(s, t, k=k))
        assert got == pytest.approx(exp)


def test_get_distance_registry_weighted_angle_binds_rho_correctly() -> None:
    reg = dr.get_distance_registry(rho_values=(0.5, 1.618), k_values=(), include_optional=False)
    s, t = "GATTACA", "GACTATA"

    for rho in (0.5, 1.618):
        key = f"weighted_angle_rho={rho}"
        assert key in reg
        got = reg[key](s, t)
        exp = float(naive_weighted_angle_distance(s, t, rho=rho, max_n=None))
        assert got == pytest.approx(exp)


def test_get_distance_registry_includes_optional_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rapidfuzz(monkeypatch, jw_value=0.12, lcs_value=3.0)
    reg = dr.get_distance_registry(rho_values=(0.5,), k_values=(2,), include_optional=True)

    assert "jaro_winkler" in reg
    assert reg["jaro_winkler"]("a", "b") == 0.12

    # Intended behavior: include only if RapidFuzz available and callable
    assert "lcs" in reg
    assert reg["lcs"]("abc", "xyz") == 3.0


def test_get_distance_registry_skips_optional_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    reg = dr.get_distance_registry(rho_values=(0.5,), k_values=(2,), include_optional=True)

    assert "jaro_winkler" not in reg
    # Intended behavior: also skip lcs if RapidFuzz missing
    assert "lcs" not in reg
