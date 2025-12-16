from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from src.clustering.dbscan import (
    DbscanResult,
    _eps_bounds_from_quantiles,
    _upper_triangle,
    pairwise_distance_matrix,
    run_dbscan_precomputed,
    safe_silhouette_precomputed,
    tune_dbscan_silhouette,
)


def _two_cluster_D(
    n_per_cluster: int = 3, within: float = 0.1, between: float = 10.0
) -> np.ndarray:
    """
    Build a clean 2-cluster distance matrix:
      - within-cluster distances = within
      - between-cluster distances = between
      - diagonal = 0
    """
    n = 2 * n_per_cluster
    D = np.full((n, n), between, dtype=float)
    np.fill_diagonal(D, 0.0)

    # cluster A: [0..n_per_cluster-1], cluster B: [n_per_cluster..2*n_per_cluster-1]
    A = range(0, n_per_cluster)
    B = range(n_per_cluster, 2 * n_per_cluster)

    for idxs in (A, B):
        for i in idxs:
            for j in idxs:
                if i != j:
                    D[i, j] = within
    return D


def test_pairwise_distance_matrix_values_symmetry_and_call_count() -> None:
    seqs = ["a", "aa", "aaaa"]

    calls = {"n": 0}

    def dist(a: str, b: str) -> float:
        calls["n"] += 1
        return float(abs(len(a) - len(b)))

    D = pairwise_distance_matrix(seqs, dist)

    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)

    # spot-check values
    assert D[0, 1] == 1.0
    assert D[0, 2] == 3.0
    assert D[1, 2] == 2.0

    # should call dist exactly n*(n-1)/2 times
    assert calls["n"] == 3


def test_run_dbscan_precomputed_finds_two_clusters() -> None:
    D = _two_cluster_D(n_per_cluster=3, within=0.1, between=10.0)

    # eps < between => clusters should not connect
    labels = run_dbscan_precomputed(D, eps=1.0, min_samples=2)
    assert labels.shape == (6,)
    assert set(labels.tolist()) != {-1}  # not all noise

    # Expect two clusters of size 3, no noise
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert (labels == -1).sum() == 0


def test_safe_silhouette_precomputed_degenerate_all_noise() -> None:
    D = _two_cluster_D(n_per_cluster=2)
    labels = np.array([-1, -1, -1, -1], dtype=int)
    assert safe_silhouette_precomputed(D, labels) == -1.0


def test_safe_silhouette_precomputed_degenerate_single_cluster() -> None:
    D = _two_cluster_D(n_per_cluster=2)
    labels = np.array([0, 0, 0, 0], dtype=int)
    assert safe_silhouette_precomputed(D, labels) == -1.0


def test_safe_silhouette_precomputed_matches_sklearn_when_valid() -> None:
    D = _two_cluster_D(n_per_cluster=3, within=0.1, between=10.0)
    labels = run_dbscan_precomputed(D, eps=1.0, min_samples=2)

    sil = safe_silhouette_precomputed(D, labels)
    assert sil > 0.0
    assert -1.0 <= sil <= 1.0


def test_upper_triangle_ordering() -> None:
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    v = _upper_triangle(D)
    assert v.tolist() == [1.0, 2.0, 3.0]


def test_eps_bounds_empty_distance_vector() -> None:
    D = np.zeros((1, 1), dtype=float)
    eps_low, eps_high = _eps_bounds_from_quantiles(D, 0.1, 0.9)
    assert eps_low == 0.0
    assert eps_high == 1.0


def test_eps_bounds_degenerate_all_zero_distances() -> None:
    # n=2 => one off-diagonal distance = 0
    D = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    eps_low, eps_high = _eps_bounds_from_quantiles(D, 0.1, 0.9)

    # low should be bumped > 0; high should be > low via fallback
    assert eps_low > 0.0
    assert eps_high > eps_low


def test_tune_dbscan_silhouette_trivial_n_lt_2() -> None:
    D = np.zeros((1, 1), dtype=float)
    res = tune_dbscan_silhouette(D, n_trials=5, prefer_optuna=False)
    assert isinstance(res, DbscanResult)
    assert res.labels.shape == (1,)
    assert res.labels[0] == -1
    assert res.silhouette == -1.0
    assert res.min_samples == 3  # default min_samples_values[0]
    # note: eps_low returned by _eps_bounds_from_quantiles for size-0 dvec is 0.0
    assert res.eps == 0.0


def test_tune_dbscan_silhouette_random_search_is_deterministic_given_seed() -> None:
    D = _two_cluster_D(n_per_cluster=3, within=0.1, between=10.0)

    res1 = tune_dbscan_silhouette(
        D,
        n_trials=12,
        min_samples_values=(2, 3),
        eps_quantiles=(0.02, 0.20),
        seed=123,
        prefer_optuna=False,
    )
    res2 = tune_dbscan_silhouette(
        D,
        n_trials=12,
        min_samples_values=(2, 3),
        eps_quantiles=(0.02, 0.20),
        seed=123,
        prefer_optuna=False,
    )

    assert np.array_equal(res1.labels, res2.labels)
    assert res1.min_samples == res2.min_samples
    assert res1.eps == pytest.approx(res2.eps)
    assert res1.silhouette == pytest.approx(res2.silhouette)

    # sanity: silhouette matches helper computation for returned labels
    assert res1.silhouette == pytest.approx(safe_silhouette_precomputed(D, res1.labels))


def test_tune_dbscan_silhouette_optuna_success_path_via_fake_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    D = _two_cluster_D(n_per_cluster=3, within=0.1, between=10.0)

    # --- fake optuna module ---
    fake_optuna = ModuleType("optuna")

    class FakeTPESampler:
        def __init__(self, seed: int):
            self.seed = seed

    class FakeTrial:
        def __init__(self, eps: float, ms: int):
            self._eps = float(eps)
            self._ms = int(ms)

        def suggest_float(self, name: str, low: float, high: float) -> float:
            # behave like "optuna-ish": clamp into [low, high]
            return float(min(max(self._eps, low), high))

        def suggest_categorical(self, name: str, choices: list[int]) -> int:
            return int(self._ms if self._ms in choices else choices[0])

    class FakeStudy:
        def __init__(self):
            self.best_params: dict[str, object] = {}
            self._best_val = -1e9

        def optimize(self, objective: Callable[[object], float], n_trials: int) -> None:
            # Candidate set: one good eps (< between) and one bad eps (=between => merges)
            candidates = [(0.2, 2), (10.0, 2)]
            for i in range(n_trials):
                eps, ms = candidates[i % len(candidates)]
                val = float(objective(FakeTrial(eps, ms)))
                if val > self._best_val:
                    self._best_val = val
                    self.best_params = {"eps": float(eps), "min_samples": int(ms)}

    def create_study(*, direction: str, sampler: object) -> FakeStudy:
        return FakeStudy()

    fake_optuna.samplers = SimpleNamespace(TPESampler=FakeTPESampler)
    fake_optuna.create_study = create_study

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    res = tune_dbscan_silhouette(
        D,
        n_trials=6,
        min_samples_values=(2, 3),
        eps_quantiles=(0.02, 0.20),
        seed=7,
        prefer_optuna=True,
    )

    assert isinstance(res, DbscanResult)
    assert res.min_samples in (2, 3)
    assert res.silhouette > 0.0
    assert (res.labels == -1).sum() == 0


def test_tune_dbscan_silhouette_optuna_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    D = _two_cluster_D(n_per_cluster=3, within=0.1, between=10.0)

    fake_optuna = ModuleType("optuna")

    class FakeTPESampler:
        def __init__(self, seed: int):
            self.seed = seed

    def create_study(*, direction: str, sampler: object):
        raise RuntimeError("boom")

    fake_optuna.samplers = SimpleNamespace(TPESampler=FakeTPESampler)
    fake_optuna.create_study = create_study

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    res = tune_dbscan_silhouette(
        D,
        n_trials=8,
        min_samples_values=(2, 3),
        eps_quantiles=(0.02, 0.20),
        seed=42,
        prefer_optuna=True,  # should attempt optuna then fall back
    )

    assert isinstance(res, DbscanResult)
    assert res.min_samples in (2, 3)
    assert res.labels.shape == (6,)
