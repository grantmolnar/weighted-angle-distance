# src/clustering/viz.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure scripts can save figures on headless machines
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def _to_int_labels(labels: Sequence[Any]) -> np.ndarray:
    """
    Map arbitrary labels to consecutive ints (stable order by first appearance).
    Useful for consistent coloring.
    """
    mapping: dict[Any, int] = {}
    out = np.empty(len(labels), dtype=int)
    next_id = 0
    for i, x in enumerate(labels):
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
        out[i] = mapping[x]
    return out


def maybe_plot_distance_heatmap(
    D: np.ndarray,
    true_labels: Sequence[Any],
    pred_labels: Sequence[int],
    out_path: str | Path,
) -> Path:
    """
    Save a distance-matrix heatmap, ordered by true labels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(_to_int_labels(true_labels), kind="stable")
    D_ord = D[np.ix_(order, order)]

    plt.figure()
    plt.imshow(D_ord, aspect="auto")
    plt.title("Distance matrix (ordered by true label)")
    plt.xlabel("items")
    plt.ylabel("items")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def maybe_plot_mds(
    D: np.ndarray,
    labels: Sequence[Any],
    out_path: str | Path,
    *,
    title: str,
    random_state: int = 0,
) -> Path:
    """
    Save a 2D MDS embedding from a precomputed distance matrix.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # MDS on precomputed distances
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=1,
        max_iter=300,
    )
    coords = mds.fit_transform(D)

    c = _to_int_labels(labels)

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], c=c, s=12)
    plt.title(title)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path
