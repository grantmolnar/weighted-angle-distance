from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering.evaluation import evaluate_against_labels


@settings(max_examples=150, deadline=None)
@given(
    st.integers(min_value=1, max_value=40).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(min_value=-3, max_value=6), min_size=n, max_size=n),
            st.lists(st.text(min_size=1, max_size=3), min_size=n, max_size=n),
        )
    )
)
def test_invariants_and_scores_match_sklearn(data: tuple[list[int], list[str]]) -> None:
    pred, true = data
    out = evaluate_against_labels(pred, true)

    pred_np = np.asarray(pred)

    # Invariants
    assert out.n_points == len(pred)
    assert out.n_noise == int((pred_np == -1).sum())

    uniq = set(pred)
    expected_clusters = len(uniq) - (1 if -1 in uniq else 0)
    assert out.n_clusters_ex_noise == expected_clusters

    # Scores are exactly sklearn’s values on the same inputs
    assert out.ari == float(adjusted_rand_score(true, pred_np))
    assert out.nmi == float(normalized_mutual_info_score(true, pred_np))


@settings(max_examples=120, deadline=None)
@given(
    st.integers(min_value=1, max_value=40).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(min_value=-2, max_value=5), min_size=n, max_size=n),
            st.lists(st.text(min_size=1, max_size=3), min_size=n, max_size=n),
            st.permutations(range(n)),
        )
    )
)
def test_permutation_invariance(data: tuple[list[int], list[str], tuple[int, ...]]) -> None:
    pred, true, perm = data

    out1 = evaluate_against_labels(pred, true)

    pred2 = [pred[i] for i in perm]
    true2 = [true[i] for i in perm]
    out2 = evaluate_against_labels(pred2, true2)

    assert out1.ari == pytest.approx(out2.ari)
    assert out1.nmi == pytest.approx(out2.nmi)
    assert out1.n_points == out2.n_points
    assert out1.n_noise == out2.n_noise
    assert out1.n_clusters_ex_noise == out2.n_clusters_ex_noise
