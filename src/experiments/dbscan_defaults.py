# src/experiments/dbscan_defaults.py
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.clustering.suite import DataImporter, DistanceSpec
from src.data.splice_loader import load_splice_to_polars
from src.string_distances.distance_registry import get_distance_registry


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


SPLICE_PATH = (
    repo_root()
    / "src"
    / "data"
    / "molecular+biology+splice+junction+gene+sequences"
    / "splice.data"
)


def load_splice() -> pl.DataFrame:
    return load_splice_to_polars(SPLICE_PATH)


DATA_IMPORTERS: list[DataImporter] = [
    DataImporter(name="splice", load=load_splice),
]


# Build the full registry once, then pick a subset by name.
_RHOS: list[float] = [0.1 * i for i in range(1, 10)]
_KS: list[int] = list(range(2, 50))

_REG = get_distance_registry(
    rho_values=_RHOS,
    k_values=_KS,
    max_n_for_weighted=60,
    include_optional=True,
)

DISTANCE_KEYS: list[str] = (
    [
        "levenshtein",
        "jaro_winkler",  # optional: may not exist
        "lcs",  # optional: may not exist
    ]
    + [f"weighted_angle_rho={r}" for r in _RHOS]
    + [f"kgram_angle_k={k}" for k in range(2, 7)]
)

# If you want to be strict about some distances existing, do it here.
_REQUIRED_KEYS: set[str] = {"levenshtein"}
missing_required = sorted(_REQUIRED_KEYS - set(_REG))
if missing_required:
    raise RuntimeError(f"Missing required distances in registry: {missing_required}")

# Avoid KeyError when optional deps aren't installed.
DISTANCES: list[DistanceSpec] = [
    DistanceSpec(name=k, fn=_REG[k]) for k in DISTANCE_KEYS if k in _REG
]
