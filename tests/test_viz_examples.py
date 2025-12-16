from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.clustering.viz import (
    _to_int_labels,
    maybe_plot_distance_heatmap,
    maybe_plot_mds,
)


def test__to_int_labels_stable_by_first_appearance() -> None:
    labels = ["cat", "dog", "cat", "emu", "dog", "fox"]
    got = _to_int_labels(labels)

    # First-seen order: cat->0, dog->1, emu->2, fox->3
    assert got.tolist() == [0, 1, 0, 2, 1, 3]


def test_maybe_plot_distance_heatmap_orders_by_true_label_and_writes_file(
    tmp_path: Path,
) -> None:
    # Make a matrix with unique entries so reordering is easy to verify.
    n = 5
    D = np.arange(n * n, dtype=float).reshape(n, n)

    true_labels = ["b", "a", "b", "c", "a"]
    pred_labels = [
        -1,
        0,
        0,
        1,
        -1,
    ]  # not used by the function, but required by signature

    captured: dict[str, np.ndarray] = {}
    orig_imshow = plt.imshow

    def imshow_hook(A, *args, **kwargs):
        captured["A"] = np.array(A, copy=True)
        return orig_imshow(A, *args, **kwargs)

    # Patch plt.imshow just to capture the matrix it receives.
    plt.imshow = imshow_hook  # type: ignore[assignment]
    try:
        out_path = tmp_path / "subdir" / "heatmap.png"
        ret = maybe_plot_distance_heatmap(D, true_labels, pred_labels, out_path)
    finally:
        plt.imshow = orig_imshow  # type: ignore[assignment]

    assert ret == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Verify stable ordering by true label (as implemented).
    order = np.argsort(_to_int_labels(true_labels), kind="stable")
    expected = D[np.ix_(order, order)]
    np.testing.assert_array_equal(captured["A"], expected)


def test_maybe_plot_mds_writes_file(tmp_path: Path) -> None:
    # Construct a valid symmetric distance matrix from points in R^2.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    labels = ["L1", "L2", "L2", "L3"]
    out_path = tmp_path / "plots" / "mds.png"

    ret = maybe_plot_mds(D, labels, out_path, title="toy", random_state=0)

    assert ret == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
