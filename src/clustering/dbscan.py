from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from src.string_distances.distance_registry import DistanceFn


def pairwise_distance_matrix(sequences, dist):
    n = len(sequences)

    # Fast path: distance function optionally exposes a vectorized pairwise builder.
    pairwise = getattr(dist, "pairwise", None)
    if callable(pairwise):
        D = np.asarray(pairwise(sequences), dtype=float)
        if D.shape != (n, n):
            raise ValueError(f"dist.pairwise returned shape {D.shape}, expected {(n, n)}")
        # Ensure exact zeros on diagonal (some implementations may not guarantee it)
        np.fill_diagonal(D, 0.0)
        return D

    # Fallback: O(N^2) calls to dist
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i % 25 == 0:
            print(f"[D] row {i}/{n}", flush=True)
        si = sequences[i]
        for j in range(i + 1, n):
            d = dist(si, sequences[j])
            D[i, j] = d
            D[j, i] = d
    return D




def run_dbscan_precomputed(
    D: np.ndarray,
    *,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """
    Run DBSCAN given a precomputed distance matrix.
    Returns labels in {-1, 0, 1, ...}.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return model.fit_predict(D)


def safe_silhouette_precomputed(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score with guards for degenerate DBSCAN outcomes.

    Returns
    -------
    float
        -1.0 if silhouette is undefined (e.g. <2 clusters among non-noise points).
    """
    mask = labels != -1  # DBSCAN noise is -1
    if mask.sum() < 2:
        return -1.0

    clustered_labels = labels[mask]
    if len(set(clustered_labels)) < 2:
        return -1.0

    D_sub = D[np.ix_(mask, mask)]
    return float(silhouette_score(D_sub, clustered_labels, metric="precomputed"))


def _upper_triangle(D: np.ndarray) -> np.ndarray:
    """Return the upper-triangle distances (excluding diagonal)."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def _eps_bounds_from_quantiles(
    D: np.ndarray, q_low: float, q_high: float
) -> tuple[float, float]:
    """
    Choose eps bounds based on quantiles of pairwise distances.

    Falls back to (min,max) if quantiles are degenerate.
    """
    dvec = _upper_triangle(D)
    if dvec.size == 0:
        return 0.0, 1.0

    eps_low = float(np.quantile(dvec, q_low))
    eps_high = float(np.quantile(dvec, q_high))

    # Validate / fallback
    if not np.isfinite(eps_low) or not np.isfinite(eps_high) or eps_high <= eps_low:
        eps_low, eps_high = float(dvec.min()), float(dvec.max())
        if eps_high <= eps_low:
            eps_high = eps_low + 1.0

    # DBSCAN with eps<=0 is basically meaningless for nonnegative distances
    if eps_low <= 0:
        eps_low = max(eps_low, 1e-12)

    # Ensure the interval is valid after bumping eps_low
    if eps_high < eps_low:
        eps_high = eps_low + max(1e-12, 1e-6 * eps_low)

    return eps_low, eps_high


@dataclass(frozen=True)
class DbscanResult:
    labels: np.ndarray
    silhouette: float
    eps: float
    min_samples: int


def tune_dbscan_silhouette(
    D: np.ndarray,
    *,
    n_trials: int = 100,
    min_samples_values: Sequence[int] = (3, 5, 8, 13),
    eps_quantiles: tuple[float, float] = (0.02, 0.20),
    seed: int = 0,
    prefer_optuna: bool = True,
) -> DbscanResult:
    """
    Tune DBSCAN hyperparameters using silhouette score only (no true labels).

    Strategy
    --------
    1) Compute eps_low/eps_high from quantiles of the pairwise-distance distribution.
    2) If Optuna is available (and prefer_optuna=True), run TPE search.
       Otherwise, do a small random search.

    Returns
    -------
    DbscanResult
        Best labels + silhouette + (eps, min_samples).
    """
    eps_low, eps_high = _eps_bounds_from_quantiles(
        D, eps_quantiles[0], eps_quantiles[1]
    )

    # Trivial / tiny case
    if D.shape[0] < 2:
        return DbscanResult(
            labels=np.full((D.shape[0],), -1, dtype=int),
            silhouette=-1.0,
            eps=eps_low,
            min_samples=int(min_samples_values[0]),
        )

    # Optuna path (optional)
    if prefer_optuna:
        try:
            import optuna

            def objective(trial: "optuna.Trial") -> float:
                eps = trial.suggest_float("eps", eps_low, eps_high)
                ms = trial.suggest_categorical("min_samples", list(min_samples_values))
                labels = run_dbscan_precomputed(D, eps=float(eps), min_samples=int(ms))
                return safe_silhouette_precomputed(D, labels)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(objective, n_trials=n_trials)

            best_eps = float(study.best_params["eps"])
            best_ms = int(study.best_params["min_samples"])
            best_labels = run_dbscan_precomputed(D, eps=best_eps, min_samples=best_ms)
            best_sil = float(safe_silhouette_precomputed(D, best_labels))
            return DbscanResult(best_labels, best_sil, best_eps, best_ms)

        except Exception:
            # fall through to random search
            pass

    # Random search fallback (no extra deps)
    rng = np.random.default_rng(seed)
    best_sil = -1.0
    best_eps = eps_low
    best_ms = int(min_samples_values[0])
    best_labels = np.full((D.shape[0],), -1, dtype=int)

    ms_choices = np.array(list(min_samples_values), dtype=int)
    for _ in range(n_trials):
        eps = float(rng.uniform(eps_low, eps_high))
        ms = int(rng.choice(ms_choices))
        labels = run_dbscan_precomputed(D, eps=eps, min_samples=ms)
        sil = float(safe_silhouette_precomputed(D, labels))
        if sil > best_sil:
            best_sil, best_eps, best_ms, best_labels = sil, eps, ms, labels

    return DbscanResult(best_labels, float(best_sil), float(best_eps), int(best_ms))
