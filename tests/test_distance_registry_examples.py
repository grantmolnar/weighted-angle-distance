import sys
from types import ModuleType

import numpy as np
import pytest

from src.string_distances import distance_registry as dr


def _clear_distance_registry_caches() -> None:
    # These exist only after the speedup patch; guard for safety.
    for name in ("_rf_levenshtein_distance", "_rf_process_cdist"):
        fn = getattr(dr, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()  # type: ignore[attr-defined]


def _install_fake_rapidfuzz(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lev_value: float = 5.0,
    lcs_value: float = 7.0,
    dam_value: float = 1.0,
    include_process: bool = True,
) -> None:
    """Install a minimal fake rapidfuzz in sys.modules for deterministic tests."""

    rf = ModuleType("rapidfuzz")
    dist = ModuleType("rapidfuzz.distance")
    lev = ModuleType("rapidfuzz.distance.Levenshtein")
    lcsseq = ModuleType("rapidfuzz.distance.LCSseq")
    dam = ModuleType("rapidfuzz.distance.DamerauLevenshtein")

    lev.distance = lambda a, b: lev_value  # type: ignore[attr-defined]
    lcsseq.distance = lambda a, b: lcs_value  # type: ignore[attr-defined]
    dam.distance = lambda a, b: dam_value  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rapidfuzz", rf)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", dist)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.Levenshtein", lev)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.LCSseq", lcsseq)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance.DamerauLevenshtein", dam)

    if include_process:
        proc = ModuleType("rapidfuzz.process")

        def _cdist(xs, ys, scorer, dtype=None):  # type: ignore[no-redef]
            out = np.zeros((len(xs), len(ys)), dtype=float)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    out[i, j] = float(scorer(x, y))
            return out

        proc.cdist = _cdist  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rapidfuzz.process", proc)

    _clear_distance_registry_caches()


def _force_rapidfuzz_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure rapidfuzz is unavailable by removing it and stubbing a blank module."""
    for key in list(sys.modules):
        if key.startswith("rapidfuzz"):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setitem(sys.modules, "rapidfuzz", ModuleType("rapidfuzz"))
    _clear_distance_registry_caches()


def test_levenshtein_distance_uses_rapidfuzz_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rapidfuzz(monkeypatch, lev_value=5.0)
    assert dr.levenshtein_distance("x", "y") == 5.0


def test_levenshtein_distance_falls_back_when_rapidfuzz_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    assert dr.levenshtein_distance("kitten", "sitting") == 3.0


def test_levenshtein_pairwise_uses_cdist_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rapidfuzz(monkeypatch, lev_value=5.0, include_process=True)
    seqs = ["a", "bb", "ccc"]
    D = dr.levenshtein_distance.pairwise(seqs)  # type: ignore[attr-defined]
    assert D.shape == (3, 3)
    # If cdist path ran, diag will also be lev_value (our stub scorer is constant).
    assert np.all(D == 5.0)


def test_levenshtein_pairwise_falls_back_when_cdist_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rapidfuzz(monkeypatch, lev_value=7.0, include_process=False)
    seqs = ["a", "bb", "ccc"]
    D = dr.levenshtein_distance.pairwise(seqs)  # type: ignore[attr-defined]
    assert D.shape == (3, 3)
    assert np.all(np.diag(D) == 0.0)
    assert np.all(D[np.triu_indices(3, 1)] == 7.0)


def test_damerau_levenshtein_distance_uses_rapidfuzz_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rapidfuzz(monkeypatch, dam_value=1.0)
    assert dr.damerau_levenshtein_distance("ab", "ba") == 1.0


def test_damerau_levenshtein_distance_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_rapidfuzz_unavailable(monkeypatch)
    # Without Damerau transpositions, Levenshtein("ab","ba") == 2.
    assert dr.damerau_levenshtein_distance("ab", "ba") == 2.0


def test_jensen_shannon_kgram_distance_smoke() -> None:
    assert dr.jensen_shannon_kgram_distance("banana", "banana", k=3) == 0.0
    assert dr.jensen_shannon_kgram_distance("banana", "bandana", k=3) > 0.0
    assert dr.jensen_shannon_kgram_distance("", "", k=3) == 0.0
    assert dr.jensen_shannon_kgram_distance("abc", "", k=3) == 1.0


def test_get_distance_registry_includes_js_kgram_keys() -> None:
    reg = dr.get_distance_registry(
        rho_values=(), k_values=(), include_optional=True, js_k_values=(3,)
    )
    assert "js_kgram_k=3" in reg
    assert isinstance(reg["js_kgram_k=3"]("banana", "bandana"), float)


def test_kgram_counts_raises_for_negative_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        dr._kgram_counts("banana", -1)


def test_kgram_angle_distance_is_registered() -> None:
    reg = dr.get_distance_registry(rho_values=(), k_values=(2,), include_optional=False)
    assert "kgram_angle_k=2" in reg
    assert reg["kgram_angle_k=2"]("abc", "abc") == 0.0


def test_weighted_angle_distance_is_registered() -> None:
    reg = dr.get_distance_registry(
        rho_values=(0.5,), k_values=(), include_optional=False
    )
    assert "weighted_angle_rho=0.5" in reg
    assert reg["weighted_angle_rho=0.5"]("abc", "abc") == 0.0
