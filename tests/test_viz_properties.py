from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from hypothesis import given, settings, strategies as st
from typing import Any
from src.clustering.viz import (
    _to_int_labels,
    maybe_plot_distance_heatmap,
    maybe_plot_mds,
)


# Hashable labels only (avoid NaN floats because NaN != NaN can be surprising).
hashable_labels = st.one_of(
    st.integers(),
    st.text(min_size=0, max_size=10),
    st.tuples(st.integers(), st.integers()),
)


@settings(max_examples=80, deadline=None)
@given(labels=st.lists(hashable_labels, min_size=0, max_size=50))
def test__to_int_labels_basic_invariants(labels) -> None:
    got = _to_int_labels(labels)

    assert got.dtype == np.dtype(int)
    assert got.shape == (len(labels),)

    # Equal labels -> equal ints.
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                assert int(got[i]) == int(got[j])

    # Values are consecutive 0..k-1 if nonempty.
    if len(labels) == 0:
        return

    uniq = []
    seen = set()
    for x in labels:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    k = len(uniq)
    assert got.min() == 0
    assert got.max() == (k - 1)
    assert set(got.tolist()) == set(range(k))

    # Stable by first appearance: first occurrence index determines the id.
    first_index: dict[Any, int] = {}
    for i, x in enumerate(labels):
        first_index.setdefault(x, i)

    for x in seen:
        i0 = first_index[x]
        assert int(got[i0]) == list(uniq).index(x)


def _euclidean_distance_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    X = rng.normal(size=(n, 2))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    # Force exact symmetry and zero diagonal (nice hygiene)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


@settings(max_examples=8, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_maybe_plot_distance_heatmap_and_mds_create_files(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    D = _euclidean_distance_matrix(rng, n)

    true_labels = [f"c{i%3}" for i in range(n)]
    pred_labels = [-1] * n
    labels_for_mds = [f"g{i%4}" for i in range(n)]

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        p1 = base / "hm" / "heat.png"
        p2 = base / "mds" / "mds.png"

        ret1 = maybe_plot_distance_heatmap(D, true_labels, pred_labels, p1)
        ret2 = maybe_plot_mds(D, labels_for_mds, p2, title="prop", random_state=0)

        assert ret1 == p1 and p1.exists() and p1.stat().st_size > 0
        assert ret2 == p2 and p2.exists() and p2.stat().st_size > 0
