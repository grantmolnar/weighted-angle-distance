from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import polars as pl


def load_fluencybank_to_polars(
    path: Union[str, Path],
    *,
    min_len: int = 0,
    max_len: int = 0,
    allowed_labels: Optional[Sequence[str]] = None,
    drop_duplicate_sequences: bool = True,
) -> pl.DataFrame:
    """
    Load a processed FluencyBank parquet (label/sample_id/sequence) into Polars.

    Parameters
    ----------
    path:
        Parquet written by scripts/download_fluencybank_data.py.
    min_len, max_len:
        Optional length filters on the 'sequence' column.
        max_len=0 means no upper bound.
    allowed_labels:
        If provided, keep only these labels.
    drop_duplicate_sequences:
        If True, drop duplicate sequences (keeping first occurrence).

    Returns
    -------
    pl.DataFrame
        Columns: ["label", "sample_id", "sequence"].
    """
    p = Path(path)
    df = pl.read_parquet(p)

    # sanity
    for col in ("label", "sample_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} in {p}, found columns={df.columns}")

    if allowed_labels is not None:
        df = df.filter(pl.col("label").is_in(list(allowed_labels)))

    if min_len > 0:
        df = df.filter(pl.col("sequence").str.len_chars() >= min_len)
    if max_len and max_len > 0:
        df = df.filter(pl.col("sequence").str.len_chars() <= max_len)

    if drop_duplicate_sequences:
        df = df.unique(subset=["sequence"], keep="first")

    return df.select(["label", "sample_id", "sequence"])
