# src/experiments/dbscan_defaults.py
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.paths import raw_dir, processed_dir

from src.clustering.suite import DataImporter, DistanceSpec
from src.data.splice_loader import load_splice_to_polars
from src.data.strseq_loader import load_strseq_alleles_to_polars

from src.string_distances.distance_registry import get_distance_registry

from src.data.synthetic_tandem_repeats import (
    DEFAULT_SYNTHETIC_TR_CONFIG,
    ensure_synthetic_tandem_repeat_dataset,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


SYNTH_TR_PATH = processed_dir() / "synthetic_tandem_repeats.parquet"
SPLICE_PATH = raw_dir() / "splice" / "splice.data"
STRSEQ_PATH = processed_dir() / "strseq_alleles.parquet"


def load_synth_tr() -> pl.DataFrame:
    return ensure_synthetic_tandem_repeat_dataset(
        SYNTH_TR_PATH,
        DEFAULT_SYNTHETIC_TR_CONFIG,
        seed_labels=42,
        seed_samples=1,
    )


def load_splice() -> pl.DataFrame:
    return load_splice_to_polars(SPLICE_PATH)


def load_strseq() -> pl.DataFrame:
    return load_strseq_alleles_to_polars(
        STRSEQ_PATH,
        min_len=20,  # tweak as you like
        max_len=600,  # tweak as you like
        only_acgt=True,
        drop_duplicate_sequences=True,
        # allowed_labels=[...],  # optionally restrict to 10 loci, etc.
    )


DATA_IMPORTERS: list[DataImporter] = [
    DataImporter(name="synthetic_tr", load=load_synth_tr),
    DataImporter(name="splice", load=load_splice),
    DataImporter(name="strseq", load=load_strseq),
]


# Build the full registry once, then pick a subset by name.
r = 1 / 10
_RHOS: list[float] = [r * i for i in range(1, int(1 / r) + 1)]
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
    + [f"kgram_angle_k={k}" for k in range(4, 5)]
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
