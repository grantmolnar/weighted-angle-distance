# src/clustering/suite.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import time

import numpy as np
import polars as pl

from src.clustering.dbscan import (
    DbscanResult,
    pairwise_distance_matrix,
    run_dbscan_precomputed,
    safe_silhouette_precomputed,
    tune_dbscan_silhouette,
)
from src.clustering.evaluation import evaluate_against_labels
from src.clustering.viz import maybe_plot_distance_heatmap, maybe_plot_mds
from src.string_distances.distance_registry import DistanceFn


# ----------------------------
# Interfaces for the suite
# ----------------------------


@dataclass(frozen=True)
class DataImporter:
    """
    A dataset importer that returns a DataFrame with at least:
      - label: ground-truth class label (string)
      - sequence: the raw string to compare (string)
    """

    name: str
    load: Callable[[], pl.DataFrame]


@dataclass(frozen=True)
class DistanceSpec:
    """A named distance function (str, str) -> float."""

    name: str
    fn: DistanceFn


@dataclass(frozen=True)
class DbscanConfig:
    """
    DBSCAN configuration. If tune=True, we choose eps/min_samples
    using silhouette score only (no true labels).
    """

    tune: bool = True
    eps: float = 5.0
    min_samples: int = 5

    # Tuning behavior
    n_trials: int = 40
    min_samples_values: tuple[int, ...] = (3, 5, 8, 13)
    eps_quantiles: tuple[float, float] = (0.02, 0.20)


def run_dbscan_suite(
    importers: Sequence[DataImporter],
    distances: Sequence[DistanceSpec],
    *,
    dbscan: DbscanConfig = DbscanConfig(),
    max_rows: int | None = 500,
    seed: int = 0,
    out_dir: str | Path | None = "outputs/dbscan_suite",
    make_plots: bool = True,
    plot_max_n: int = 600,
) -> pl.DataFrame:
    """
    Run DBSCAN clustering for each (dataset importer × distance).

    Produces a Polars DataFrame of results and (optionally) plot artifacts on disk.

    Notes
    -----
    This is O(N^2) per distance due to the full pairwise distance matrix.
    Start small (e.g. max_rows=200..800) until you optimize distance computation.
    """
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for importer in importers:
        df = importer.load()

        # Optional subsample for speed / sanity.
        if max_rows is not None and df.height > max_rows:
            df = df.sample(n=max_rows, with_replacement=False, seed=seed)

        if "label" not in df.columns or "sequence" not in df.columns:
            raise ValueError(
                f"Importer {importer.name!r} must return columns ['label','sequence'] "
                f"(got columns={df.columns})."
            )

        sequences = df["sequence"].to_list()
        true_labels = df["label"].to_list()

        for dist in distances:
            t0 = time.perf_counter()
            D = pairwise_distance_matrix(sequences, dist.fn)
            t_dist = time.perf_counter() - t0

            if dbscan.tune:
                tuned: DbscanResult = tune_dbscan_silhouette(
                    D,
                    n_trials=dbscan.n_trials,
                    min_samples_values=dbscan.min_samples_values,
                    eps_quantiles=dbscan.eps_quantiles,
                    seed=seed,
                    prefer_optuna=True,
                )
                labels = tuned.labels
                sil = tuned.silhouette
                eps = tuned.eps
                min_samples = tuned.min_samples
            else:
                eps, min_samples = dbscan.eps, dbscan.min_samples
                labels = run_dbscan_precomputed(D, eps=eps, min_samples=min_samples)
                sil = float(safe_silhouette_precomputed(D, labels))

            ext = evaluate_against_labels(labels, true_labels)

            # Save plots if appropriate.
            heatmap_path = None
            mds_true_path = None
            mds_pred_path = None
            if make_plots and out_path is not None and len(sequences) <= plot_max_n:
                tag = f"{importer.name}__{dist.name}".replace("/", "_")
                heatmap_path = maybe_plot_distance_heatmap(
                    D, true_labels, labels, out_path / f"{tag}__heatmap.png"
                )
                mds_true_path = maybe_plot_mds(
                    D,
                    true_labels,
                    out_path / f"{tag}__mds_true.png",
                    title=f"{tag} (true labels)",
                )
                mds_pred_path = maybe_plot_mds(
                    D,
                    labels.tolist(),
                    out_path / f"{tag}__mds_pred.png",
                    title=f"{tag} (DBSCAN labels)",
                )

            rows.append(
                {
                    "dataset": importer.name,
                    "distance": dist.name,
                    "n": len(sequences),
                    "t_pairwise_seconds": t_dist,
                    "tuned": dbscan.tune,
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "silhouette": float(sil),
                    "ari": float(ext.ari),
                    "nmi": float(ext.nmi),
                    "n_clusters_ex_noise": int(ext.n_clusters_ex_noise),
                    "n_noise": int(ext.n_noise),
                    "heatmap_path": str(heatmap_path) if heatmap_path else None,
                    "mds_true_path": str(mds_true_path) if mds_true_path else None,
                    "mds_pred_path": str(mds_pred_path) if mds_pred_path else None,
                }
            )

    result_df = pl.DataFrame(rows)

    if out_path is not None:
        result_df.write_parquet(out_path / "results.parquet")
        result_df.write_csv(out_path / "results.csv")

    return result_df
