from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import polars as pl


def _only_acgt_expr(col: str = "sequence") -> pl.Expr:
    # fast-ish ACGT check via regex
    return pl.col(col).str.contains(r"^[ACGT]+$")


def load_ucsc_trf_to_polars(
    path: Union[str, Path],
    *,
    min_len: int = 20,
    max_len: int = 600,
    only_acgt: bool = True,
    drop_duplicate_sequences: bool = True,
    allowed_labels: Optional[Iterable[str]] = None,
    max_samples_per_label: Optional[int] = None,
    seed: int = 0,
) -> pl.DataFrame:
    """
    Load a parquet built by scripts.download_ucsc_trf_repeats into a Polars DataFrame.

    Expected columns: ["label", "sample_id", "sequence"].

    Parameters
    ----------
    path:
        Parquet path.
    min_len, max_len:
        Length filters on 'sequence'.
    only_acgt:
        Keep only A/C/G/T sequences.
    drop_duplicate_sequences:
        Drop exact duplicate sequences (keeps first).
    allowed_labels:
        Optional label whitelist (repeat units).
    max_samples_per_label:
        Optional cap per label (downsample deterministically with seed).
    seed:
        RNG seed for per-label downsampling.

    Returns
    -------
    pl.DataFrame with columns ["label", "sample_id", "sequence"] (and any extra columns preserved).
    """
    p = Path(path)
    df = pl.read_parquet(p)

    # Basic sanity
    required = {"label", "sample_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {p}: {sorted(missing)}")

    df = df.with_columns(
        pl.col("label").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("sample_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("sequence").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
    )

    df = df.filter(pl.col("sequence").str.len_chars().is_between(min_len, max_len))

    if only_acgt:
        df = df.filter(_only_acgt_expr("sequence") & _only_acgt_expr("label"))

    if allowed_labels is not None:
        allowed = [str(x).strip().upper() for x in allowed_labels]
        df = df.filter(pl.col("label").is_in(allowed))

    if drop_duplicate_sequences:
        df = df.unique(subset=["sequence"], keep="first")

    if max_samples_per_label is not None:
        # deterministic per-label sampling
        df = (
            df.with_columns(pl.int_range(0, pl.len()).over("label").alias("_row_ix"))
            .with_columns(pl.col("_row_ix").shuffle(seed=seed).over("label").alias("_shuf"))
            .filter(pl.col("_shuf") < max_samples_per_label)
            .drop(["_row_ix", "_shuf"])
        )

    return df.select(["label", "sample_id", "sequence"])
