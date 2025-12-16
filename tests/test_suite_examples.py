from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, cast

import numpy as np
import polars as pl
import pytest

import src.clustering.suite as suite_mod
from src.clustering.dbscan import DbscanResult
from src.clustering.evaluation import ExternalMetrics


def _df(labels: list[str], seqs: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"label": labels, "sequence": seqs})


def test_run_dbscan_suite_subsamples_when_max_rows_smaller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = suite_mod.DataImporter(
        name="big",
        load=lambda: _df(["a"] * 20, [f"s{i}" for i in range(20)]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    def fake_pairwise_distance_matrix(
        seqs: Sequence[str],
        fn: Callable[[str, str], float],
    ) -> np.ndarray:
        # Cast to unparameterized np.ndarray to avoid dtype-invariance mypy issues.
        return cast(np.ndarray, np.zeros((len(seqs), len(seqs)), dtype=float))

    def fake_run_dbscan_precomputed(
        D: np.ndarray, *, eps: float, min_samples: int
    ) -> np.ndarray:
        return np.zeros((D.shape[0],), dtype=int)

    def fake_safe_silhouette_precomputed(D: np.ndarray, labels: np.ndarray) -> float:
        return -1.0

    def fake_evaluate_against_labels(
        pred_labels: Sequence[int],
        true_labels: Sequence[str],
    ) -> ExternalMetrics:
        return ExternalMetrics(
            ari=0.0,
            nmi=0.0,
            n_points=len(list(pred_labels)),
            n_noise=0,
            n_clusters_ex_noise=1,
        )

    monkeypatch.setattr(
        suite_mod, "pairwise_distance_matrix", fake_pairwise_distance_matrix
    )
    monkeypatch.setattr(
        suite_mod, "run_dbscan_precomputed", fake_run_dbscan_precomputed
    )
    monkeypatch.setattr(
        suite_mod, "safe_silhouette_precomputed", fake_safe_silhouette_precomputed
    )
    monkeypatch.setattr(
        suite_mod, "evaluate_against_labels", fake_evaluate_against_labels
    )

    res = suite_mod.run_dbscan_suite(
        [importer],
        [dist],
        dbscan=suite_mod.DbscanConfig(tune=False),
        max_rows=7,
        seed=123,
        out_dir=None,
        make_plots=False,
    )

    assert res.height == 1
    assert res["n"][0] == 7
