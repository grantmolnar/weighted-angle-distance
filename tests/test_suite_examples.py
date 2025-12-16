from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
import pytest

import src.clustering.suite as suite_mod
from src.clustering.dbscan import DbscanResult
from src.clustering.evaluation import ExternalMetrics


def _df(labels: Sequence[str], seqs: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame({"label": list(labels), "sequence": list(seqs)})


def test_run_dbscan_suite_tuned_with_plots_and_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # perf_counter called twice per distance (start/end)
    times = iter([100.0, 101.25])  # => 1.25 seconds
    monkeypatch.setattr(suite_mod.time, "perf_counter", lambda: next(times))

    def fake_pairwise_distance_matrix(
        sequences: Sequence[str],
        dist_fn: Callable[[str, str], float],
    ) -> np.ndarray:
        n = len(sequences)
        D = np.ones((n, n), dtype=float)
        np.fill_diagonal(D, 0.0)
        return D

    monkeypatch.setattr(
        suite_mod, "pairwise_distance_matrix", fake_pairwise_distance_matrix
    )

    tuned_out = DbscanResult(
        labels=np.array([0, 0, 1, -1], dtype=int),
        silhouette=0.42,
        eps=0.123,
        min_samples=5,
    )

    def fake_tune_dbscan_silhouette(
        D: np.ndarray,
        *,
        n_trials: int,
        min_samples_values: Sequence[int],
        eps_quantiles: tuple[float, float],
        seed: int,
        prefer_optuna: bool,
    ) -> DbscanResult:
        assert isinstance(D, np.ndarray)
        assert prefer_optuna is True
        assert isinstance(seed, int)
        return tuned_out

    monkeypatch.setattr(
        suite_mod, "tune_dbscan_silhouette", fake_tune_dbscan_silhouette
    )

    def fake_evaluate_against_labels(
        pred_labels: Sequence[int],
        true_labels: Sequence[str],
    ) -> ExternalMetrics:
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

    # plotters: return Path for heatmap + true MDS; return None for pred MDS (covers both conversions)
    def fake_heatmap(
        D: np.ndarray,
        true_labels: Sequence[Any],
        pred_labels: Sequence[int],
        out_path: Path,
    ) -> Path:
        # ensure "/" was sanitized in tag
        assert "A_B" in out_path.name
        assert "dist_X" in out_path.name
        out_path.write_text("heatmap")
        return out_path

    def fake_mds(
        D: np.ndarray,
        labels: Sequence[Any],
        out_path: Path,
        *,
        title: str,
        random_state: int = 0,
    ) -> Path | None:
        if "DBSCAN labels" in title:
            return None
        out_path.write_text("mds")
        return out_path

    monkeypatch.setattr(suite_mod, "maybe_plot_distance_heatmap", fake_heatmap)
    monkeypatch.setattr(suite_mod, "maybe_plot_mds", fake_mds)

    # patch polars writes to avoid parquet engine deps and still execute the write lines
    def fake_write_parquet(self: pl.DataFrame, path: object) -> None:
        Path(str(path)).write_text("parquet")

    def fake_write_csv(self: pl.DataFrame, path: object) -> None:
        Path(str(path)).write_text("csv")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write_parquet, raising=True)
    monkeypatch.setattr(pl.DataFrame, "write_csv", fake_write_csv, raising=True)

    importer = suite_mod.DataImporter(
        name="A/B",
        load=lambda: _df(["x", "x", "y", "y"], ["aa", "ab", "ba", "bb"]),
    )
    dist = suite_mod.DistanceSpec(name="dist/X", fn=lambda a, b: 0.0)

    out_dir = tmp_path / "out"
    res = suite_mod.run_dbscan_suite(
        importers=[importer],
        distances=[dist],
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

    assert isinstance(row["heatmap_path"], str) and row["heatmap_path"].endswith(".png")
    assert isinstance(row["mds_true_path"], str) and row["mds_true_path"].endswith(
        ".png"
    )
    assert row["mds_pred_path"] is None  # by fake_mds

    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "results.csv").exists()


def test_run_dbscan_suite_not_tuned_out_dir_none_skips_plots_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure plotters are never called (out_dir=None short-circuits plot branch)
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_distance_heatmap",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("heatmap should not be called")
        ),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_mds",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("mds should not be called")
        ),
    )

    def fake_pairwise_distance_matrix(
        sequences: Sequence[str],
        dist_fn: Callable[[str, str], float],
    ) -> np.ndarray:
        n = len(sequences)
        return np.zeros((n, n), dtype=float)

    monkeypatch.setattr(
        suite_mod, "pairwise_distance_matrix", fake_pairwise_distance_matrix
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
        lambda pred_labels, true_labels: ExternalMetrics(
            ari=0.0,
            nmi=0.0,
            n_points=len(pred_labels),
            n_noise=0,
            n_clusters_ex_noise=1,
        ),
    )

    importer = suite_mod.DataImporter(
        name="tiny",
        load=lambda: _df(["a", "b", "c"], ["x", "y", "z"]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    res = suite_mod.run_dbscan_suite(
        importers=[importer],
        distances=[dist],
        dbscan=suite_mod.DbscanConfig(tune=False, eps=9.0, min_samples=2),
        max_rows=None,
        seed=0,
        out_dir=None,
        make_plots=True,
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


def test_run_dbscan_suite_plot_gate_len_exceeds_plot_max_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # out_dir exists but len(seqs) > plot_max_n => skip plotting branch
    monkeypatch.setattr(
        suite_mod,
        "pairwise_distance_matrix",
        lambda seqs, fn: np.zeros((len(seqs), len(seqs)), dtype=float),
    )
    monkeypatch.setattr(
        suite_mod,
        "tune_dbscan_silhouette",
        lambda D, **k: DbscanResult(np.zeros(D.shape[0], dtype=int), -1.0, 0.1, 3),
    )
    monkeypatch.setattr(
        suite_mod,
        "evaluate_against_labels",
        lambda pred_labels, true_labels: ExternalMetrics(
            ari=0.0,
            nmi=0.0,
            n_points=len(pred_labels),
            n_noise=0,
            n_clusters_ex_noise=1,
        ),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_distance_heatmap",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not plot")),
    )
    monkeypatch.setattr(
        suite_mod,
        "maybe_plot_mds",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not plot")),
    )

    # patch writes so we execute the write lines without parquet deps
    monkeypatch.setattr(
        pl.DataFrame,
        "write_parquet",
        lambda self, path: Path(str(path)).write_text("p"),
        raising=True,
    )
    monkeypatch.setattr(
        pl.DataFrame,
        "write_csv",
        lambda self, path: Path(str(path)).write_text("c"),
        raising=True,
    )

    importer = suite_mod.DataImporter(
        name="x",
        load=lambda: _df(["a", "a", "b"], ["s1", "s2", "s3"]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    out_dir = tmp_path / "out"
    res = suite_mod.run_dbscan_suite(
        importers=[importer],
        distances=[dist],
        out_dir=out_dir,
        make_plots=True,
        plot_max_n=2,  # sequences=3 exceeds => no plot calls
    )

    row = res.row(0, named=True)
    assert row["heatmap_path"] is None
    assert row["mds_true_path"] is None
    assert row["mds_pred_path"] is None
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "results.csv").exists()


def test_run_dbscan_suite_subsamples_when_max_rows_smaller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = suite_mod.DataImporter(
        name="big",
        load=lambda: _df(["a"] * 20, [f"s{i}" for i in range(20)]),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    monkeypatch.setattr(
        suite_mod,
        "pairwise_distance_matrix",
        lambda seqs, fn: np.zeros((len(seqs), len(seqs)), dtype=float),
    )
    monkeypatch.setattr(
        suite_mod,
        "run_dbscan_precomputed",
        lambda D, *, eps, min_samples: np.zeros((D.shape[0],), dtype=int),
    )
    monkeypatch.setattr(
        suite_mod, "safe_silhouette_precomputed", lambda D, labels: -1.0
    )
    monkeypatch.setattr(
        suite_mod,
        "evaluate_against_labels",
        lambda pred_labels, true_labels: ExternalMetrics(
            ari=0.0,
            nmi=0.0,
            n_points=len(pred_labels),
            n_noise=0,
            n_clusters_ex_noise=1,
        ),
    )

    res = suite_mod.run_dbscan_suite(
        importers=[importer],
        distances=[dist],
        dbscan=suite_mod.DbscanConfig(tune=False),
        max_rows=7,
        seed=123,
        out_dir=None,
        make_plots=False,
    )

    assert res.height == 1
    assert int(res["n"][0]) == 7


def test_run_dbscan_suite_missing_required_columns_raises() -> None:
    importer = suite_mod.DataImporter(
        name="bad",
        load=lambda: pl.DataFrame({"label": ["a", "b"], "oops": ["x", "y"]}),
    )
    dist = suite_mod.DistanceSpec(name="d", fn=lambda a, b: 0.0)

    with pytest.raises(ValueError):
        suite_mod.run_dbscan_suite([importer], [dist], out_dir=None, make_plots=False)
