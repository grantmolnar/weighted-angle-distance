from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clustering.dbscan import (
    _eps_bounds_from_quantiles,
    _upper_triangle,
    pairwise_distance_matrix,
    safe_silhouette_precomputed,
    tune_dbscan_silhouette,
)


def _make_symmetric_from_upper(n: int, upper_vals: list[float]) -> np.ndarray:
    """
    Fill a symmetric n×n matrix (diag 0) using values for i<j
    in the same order as np.triu_indices(n, k=1).
    """
    D = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, k=1)
    assert len(upper_vals) == len(iu[0])
    for k, (i, j) in enumerate(zip(iu[0], iu[1], strict=True)):
        D[i, j] = float(upper_vals[k])
        D[j, i] = float(upper_vals[k])
    return D


@settings(max_examples=80)
@given(st.lists(st.text(min_size=0, max_size=8), min_size=0, max_size=10))
def test_pairwise_distance_matrix_basic_properties(seqs: list[str]) -> None:
    dist = lambda a, b: float(abs(len(a) - len(b)))  # symmetric, nonnegative
    D = pairwise_distance_matrix(seqs, dist)

    n = len(seqs)
    assert D.shape == (n, n)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)

    # spot-check all entries against definition
    for i in range(n):
        for j in range(n):
            if i == j:
                assert D[i, j] == 0.0
            else:
                assert D[i, j] == float(abs(len(seqs[i]) - len(seqs[j])))


@settings(max_examples=80)
@given(st.integers(min_value=0, max_value=8))
def test_upper_triangle_length_and_values(n: int) -> None:
    m = n * (n - 1) // 2
    upper_vals = [float(k) for k in range(m)]
    D = _make_symmetric_from_upper(n, upper_vals)
    v = _upper_triangle(D)
    assert v.shape == (m,)
    assert v.tolist() == upper_vals


@settings(max_examples=80)
@given(
    st.integers(min_value=2, max_value=8).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=n * (n - 1) // 2,
                max_size=n * (n - 1) // 2,
            ),
        )
    )
)
def test_eps_bounds_from_quantiles_returns_positive_ordered_bounds(data: tuple[int, list[float]]) -> None:
    n, upper_vals = data
    D = _make_symmetric_from_upper(n, upper_vals)

    eps_low, eps_high = _eps_bounds_from_quantiles(D, 0.10, 0.90)

    # For n>=2, function should ensure eps_low>0 and eps_high>=eps_low
    assert np.isfinite(eps_low)
    assert np.isfinite(eps_high)
    assert eps_low > 0.0
    assert eps_high >= eps_low


@settings(max_examples=60)
@given(
    st.integers(min_value=4, max_value=8).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(
                st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n * (n - 1) // 2,
                max_size=n * (n - 1) // 2,
            ),
            st.integers(min_value=2, max_value=n - 2),
        )
    )
)
def test_safe_silhouette_precomputed_is_in_range(data: tuple[int, list[float], int]) -> None:
    n, upper_vals, split = data
    D = _make_symmetric_from_upper(n, upper_vals)

    # two clusters, each size >=2
    labels = np.array([0] * split + [1] * (n - split), dtype=int)

    sil = safe_silhouette_precomputed(D, labels)
    assert -1.0 <= sil <= 1.0


@settings(max_examples=40)
@given(
    st.integers(min_value=2, max_value=8).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=n * (n - 1) // 2,
                max_size=n * (n - 1) // 2,
            ),
            st.integers(min_value=0, max_value=10_000),
        )
    )
)
def test_tune_dbscan_silhouette_random_search_basic_invariants(data: tuple[int, list[float], int]) -> None:
    n, upper_vals, seed = data
    D = _make_symmetric_from_upper(n, upper_vals)

    res = tune_dbscan_silhouette(
        D,
        n_trials=5,
        min_samples_values=(2, 3),
        eps_quantiles=(0.10, 0.90),
        seed=seed,
        prefer_optuna=False,  # keep fast + deterministic
    )

    assert res.labels.shape == (n,)
    assert res.min_samples in (2, 3)

    eps_low, eps_high = _eps_bounds_from_quantiles(D, 0.10, 0.90)
    # random search samples eps ~ Uniform(eps_low, eps_high)
    assert eps_low <= res.eps <= eps_high
