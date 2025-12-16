from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering.evaluation import ExternalMetrics, evaluate_against_labels


def test_evaluate_against_labels_no_noise_perfect_match() -> None:
    true = ["A", "A", "B", "B"]
    pred = [0, 0, 1, 1]

    out = evaluate_against_labels(pred, true)
    assert isinstance(out, ExternalMetrics)

    assert out.n_points == 4
    assert out.n_noise == 0
    assert out.n_clusters_ex_noise == 2

    # Perfect agreement => both are 1.0
    assert out.ari == 1.0
    assert out.nmi == 1.0


def test_evaluate_against_labels_with_noise_counts_and_scores_match_sklearn() -> None:
    true = ["A", "A", "B", "B", "B"]
    pred = [0, 0, -1, 1, 1]  # includes DBSCAN-style noise

    out = evaluate_against_labels(pred, true)

    assert out.n_points == 5
    assert out.n_noise == 1
    assert out.n_clusters_ex_noise == 2  # clusters {0,1}, exclude -1

    # Verify ARI/NMI exactly match sklearn on the same inputs
    pred_np = np.asarray(pred)
    assert out.ari == float(adjusted_rand_score(true, pred_np))
    assert out.nmi == float(normalized_mutual_info_score(true, pred_np))


def test_evaluate_against_labels_all_noise() -> None:
    true = ["A", "B", "C"]
    pred = [-1, -1, -1]

    out = evaluate_against_labels(pred, true)
    assert out.n_points == 3
    assert out.n_noise == 3
    assert out.n_clusters_ex_noise == 0

    pred_np = np.asarray(pred)
    assert out.ari == float(adjusted_rand_score(true, pred_np))
    assert out.nmi == float(normalized_mutual_info_score(true, pred_np))


def test_evaluate_against_labels_length_mismatch_raises_value_error() -> None:
    true = ["A", "B"]
    pred = [0, -1, 1]

    with pytest.raises(ValueError):
        evaluate_against_labels(pred, true)
