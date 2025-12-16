from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True)
class ExternalMetrics:
    ari: float
    nmi: float
    n_points: int
    n_noise: int
    n_clusters_ex_noise: int


def evaluate_against_labels(
    pred_labels: Sequence[int],
    true_labels: Sequence[str],
) -> ExternalMetrics:
    """
    Compare clustering labels to ground truth labels.

    Notes
    -----
    DBSCAN uses -1 for noise. We report:
      - ARI and NMI on all points (including noise as its own label)
      - plus simple cluster/noise counts
    """
    pred = np.asarray(pred_labels)
    n_noise = int((pred == -1).sum())
    n_clusters = len(set(pred)) - (1 if -1 in set(pred) else 0)

    ari = float(adjusted_rand_score(true_labels, pred))
    nmi = float(normalized_mutual_info_score(true_labels, pred))

    return ExternalMetrics(
        ari=ari,
        nmi=nmi,
        n_points=len(pred),
        n_noise=n_noise,
        n_clusters_ex_noise=n_clusters,
    )
