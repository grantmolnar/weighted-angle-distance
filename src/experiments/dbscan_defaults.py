# src/experiments/dbscan_defaults.py
from __future__ import annotations

from pathlib import Path

from src.clustering.suite import DataImporter, DistanceSpec
from src.data.splice_loader import load_splice_to_polars
from src.string_distances.distance_registry import get_distance_registry


def repo_root() -> Path:
    # This file lives at: <repo>/src/experiments/dbscan_defaults.py
    return Path(__file__).resolve().parents[2]


# ----------------------------
# Data sources (hardcoded)
# ----------------------------

SPLICE_PATH = (
    repo_root()
    / "src"
    / "data"
    / "molecular+biology+splice+junction+gene+sequences"
    / "splice.data"
)

DATA_IMPORTERS: list[DataImporter] = [
    DataImporter(
        name="splice",
        load=lambda p=SPLICE_PATH: load_splice_to_polars(p),
    ),
    # Add more datasets here in the same style.
]


# ----------------------------
# Distances/divergences (hardcoded)
# ----------------------------

# Build the full registry once, then pick a subset by name.
_REG = get_distance_registry(
    rho_values=(0.5,(1+5**0.5)/2),
    k_values=(2, 3, 4),
    max_n_for_weighted=60,
    include_optional=True,
)

DISTANCE_KEYS: list[str] = [
    # Pick the ones you actually want to compare:
    "levenshtein",
    "jaro_winkler",
    "weighted_angle_rho=0.5",  # adjust to match your registry key exactly
]

DISTANCES: list[DistanceSpec] = [DistanceSpec(name=k, fn=_REG[k]) for k in DISTANCE_KEYS]
