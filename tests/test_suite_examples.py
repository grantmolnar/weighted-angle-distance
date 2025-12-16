from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import src.clustering.suite as suite_mod
from src.clustering.dbscan import DbscanResult
from src.clustering.evaluation import ExternalMetrics


def _df(labels: list[str], seqs: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"label": labels, "sequence": seqs})


def test_run_dbscan_suite_tuned_with_plots_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --- patch perf_counter to make t_pairwise_seconds deterministic
    times = iter([100.0, 101.25])  # duration = 1.25
    monkeypatch.setattr(suite_mod.time, "perf_counter", lambda: next(times))

    # --- patch distance computation
    def fake_pairwise_distance_matrix(sequences: list[str], _dist_fn) -> np.ndarray:
        n = len(sequences)
        D = np.ones((n, n), dtype=float)
        np.fill_diagonal(D, 0.0)
        return D

    monkeypatch.setattr(
        suite_mod, "pairwise_distance_matrix", fake_pairwise_distance_matrix
    )

    # --- patch tuning output
    tuned_out = DbscanResult(
        labels=np.array([0, 0, 1, -1], dtype=int),
        silhouette=0.42,
        eps=0.123,
        min_samples=5,
    )

    def fake_tune_dbscan_silhouette(D: np.ndarray, **kwargs) -> DbscanResult:
        # exercise that kwargs are plumbed through
        assert "seed" in kwargs
        assert kwargs["prefer_optuna"] is True
        return tuned_out

    monkeypatch.setattr(
        suite_mod, "tune_dbscan_silhouette", fake_tune_dbscan_silhouette
    )

    # --- patch evaluation metrics (keep simple + deterministic)
    def fake_evaluate_against_labels(pred_labels, true_labels) -> ExternalMetrics:
        pred = np.asarray(pred_labels)
        return ExternalMetrics(
            ari=0.9,
            nmi=0.8,
            n_points=len(pred),
            n_noise=int((pred == -1).sum()),
            n_clusters_ex_noise=len(set(pred)) - (1 if -1 in set(pred) else 0),
        )

    monkeypatch.setattr(
        suite_mod, "evaluate_against_labels", fake_evaluate_against_labels
    )

    # --- patch plotters to cover both "Path" and "None" returns
    def fake_heatmap(D, true_labels, labels, path: Path):
        # confirm "/" replacement happened in the filename tag
        assert "A_B" in path.name  # importer has "A/B" below => "A_B"
        assert "dist_X" in path.name  # distance has "dist/X" below => "dist_X"
        path.write_text("heatmap")
        return path

    def fake_mds(D, labels, path: Path, *, title: str):
        # Return None for the true-label plot, Path for the pred-label plot
        if "true labels" in title:
            return None
        path.write_text("mds")
        return path

    monkeypatch.setattr(suite_mod, "maybe_plot_distance_heatmap", fake_heatmap)
    monkeypatch.setattr(suite_mod, "maybe_plot_mds", fake_mds)

    # --- patch polars writes to avoid parquet/csv engine dependencies
    def fake_write_parquet(self: pl.DataFrame, path) -> None:
        Path(path).write_text("parquet")

    def fake_write_csv(self: pl.DataFrame, path) -> None:
        Path(path).write_text("csv")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write_parquet, raising=True)
    monkeypatch.setattr(pl.DataFrame, "write_csv", fake_write_csv, raising=True)

    importer = suite_mod.DataImporter(
        name="A/B",
        load=lambda: _df(["x", "x", "y", "y"], ["aa", "ab", "ba", "bb"]),
    )
    dist = suite_mod.DistanceSpec(name="dist/X", fn=lambda a, b: 0.0)

    out_dir = tmp_path / "out"
    res = suite_mod.run_dbscan_suite(
        [importer],
        [dist],
        dbscan=suite_mod.DbscanConfig(tune=True),
        max_rows=None,
        seed=7,
        out_dir=out_dir,
        make_plots=True,
        plot_max_n=10,
    )

    assert res.height == 1
    row = res.row(0, named=True)
    assert row["dataset"] == "A/B"
    assert row["distance"] == "dist/X"
    assert row["n"] == 4
    assert row["tuned"] is True
    assert row["t_pairwise_seconds"] == pytest.approx(1.25)
    assert row["eps"] == pytest.approx(0.123)
    assert row["min_samples"] == 5
    assert row["silhouette"] == pytest.approx(0.42)
    assert row["ari"] == pytest.approx(0.9)
    assert row["nmi"] == pytest.approx(0.8)
    assert row["n_noise"] == 1
    assert row["n_clusters_ex_noise"] == 2

    # Plot paths: heatmap + pred mds are non-null, true mds is null (by fake_mds behavior)
    assert isinstance(row["heatmap_path"], str) and row["heatmap_path"].endswith(".png")
    assert row["mds_true_path"] is None
    assert isinstance(row["mds_pred_path"], str) and row["mds_pred_path"].endswith(
        ".png"
    )

    # Outputs written
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "results.csv").exists()


def test_run_dbscan_suite_not_tuned_no_out_dir_no_plots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # out_dir=None means: no mkdir, no plotting, no writes
    # ensure we do not accidentally call plotters
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_distance_heatmap",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_mds",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
    )

    monkeypatch.setattr(
        suite_mod,
        "pairwise_distance_matrix",
        lambda seqs, fn: np.zeros((len(seqs), len(seqs)), dtype=float),
    )
    monkeypatch.setattr(
        suite_mod,
        "run_dbscan_precomputed",
        lambda D, *, eps, min_samples: np.array([0] * D.shape[0], dtype=int),
    )
    monkeypatch.setattr(
        suite_mod, "safe_silhouette_precomputed", lambda D, labels: -1.0
    )
    monkeypatch.setattr(
        suite_mod,
        "evaluate_against_labels",
        lambda labels, true: ExternalMetrics(
            ari=0.0, nmi=0.0, n_points=len(labels), n_noise=0, n_clusters_ex_noise=1
        ),
    )

    importer = suite_mod.DataImporter(
        name="tiny",
        load=lambda: _df(["a", "b", "c"], ["x", "y", "z"]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    res = suite_mod.run_dbscan_suite(
        [importer],
        [dist],
        dbscan=suite_mod.DbscanConfig(tune=False, eps=9.0, min_samples=2),
        max_rows=None,
        seed=0,
        out_dir=None,
        make_plots=True,  # still should not plot because out_dir=None
        plot_max_n=999,
    )

    assert res.height == 1
    row = res.row(0, named=True)
    assert row["tuned"] is False
    assert row["eps"] == pytest.approx(9.0)
    assert row["min_samples"] == 2
    assert row["silhouette"] == pytest.approx(-1.0)
    assert row["heatmap_path"] is None
    assert row["mds_true_path"] is None
    assert row["mds_pred_path"] is None


def test_run_dbscan_suite_subsamples_when_max_rows_smaller(tmp_path: Path) -> None:
    # This one uses real polars.sample() to hit the subsampling branch.
    importer = suite_mod.DataImporter(
        name="big",
        load=lambda: _df(["a"] * 20, [f"s{i}" for i in range(20)]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    # Patch heavy parts but leave sample() alone
    import src.clustering.suite as suite_mod2

    suite_mod2.pairwise_distance_matrix = lambda seqs, fn: np.zeros(
        (len(seqs), len(seqs))
    )
    suite_mod2.run_dbscan_precomputed = lambda D, *, eps, min_samples: np.zeros(
        (D.shape[0],), dtype=int
    )
    suite_mod2.safe_silhouette_precomputed = lambda D, labels: -1.0
    suite_mod2.evaluate_against_labels = lambda labels, true: ExternalMetrics(
        ari=0.0, nmi=0.0, n_points=len(labels), n_noise=0, n_clusters_ex_noise=1
    )

    res = suite_mod2.run_dbscan_suite(
        [importer],
        [dist],
        dbscan=suite_mod2.DbscanConfig(tune=False),
        max_rows=7,
        seed=123,
        out_dir=None,
        make_plots=False,
    )

    assert res.height == 1
    assert res["n"][0] == 7


def test_run_dbscan_suite_missing_required_columns_raises() -> None:
    importer = suite_mod.DataImporter(
        name="bad",
        load=lambda: pl.DataFrame({"label": ["a", "b"], "oops": ["x", "y"]}),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    with pytest.raises(ValueError):
        suite_mod.run_dbscan_suite([importer], [dist], out_dir=None, make_plots=False)


def test_run_dbscan_suite_plot_gate_len_exceeds_plot_max_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # out_dir exists but len(seqs) > plot_max_n => skip plot branch
    monkeypatch.setattr(
        suite_mod,
        "pairwise_distance_matrix",
        lambda seqs, fn: np.zeros((len(seqs), len(seqs))),
    )
    monkeypatch.setattr(
        suite_mod,
        "tune_dbscan_silhouette",
        lambda D, **k: DbscanResult(np.zeros(D.shape[0], dtype=int), -1.0, 0.1, 3),
    )
    monkeypatch.setattr(
        suite_mod,
        "evaluate_against_labels",
        lambda labels, true: ExternalMetrics(
            ari=0.0, nmi=0.0, n_points=len(labels), n_noise=0, n_clusters_ex_noise=1
        ),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_distance_heatmap",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_mds",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
    )

    importer = suite_mod.DataImporter(
        name="x",
        load=lambda: _df(["a", "a", "b"], ["s1", "s2", "s3"]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    res = suite_mod.run_dbscan_suite(
        [importer],
        [dist],
        out_dir=tmp_path / "out",
        make_plots=True,
        plot_max_n=2,  # sequences=3 exceeds => no plot calls
    )
    row = res.row(0, named=True)
    assert row["heatmap_path"] is None
    assert row["mds_true_path"] is None
    assert row["mds_pred_path"] is None
