# src/data/strseq_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import polars as pl


def load_strseq_alleles_to_polars(
    path: str | Path,
    *,
    min_len: int = 1,
    max_len: Optional[int] = None,
    only_acgt: bool = True,
    drop_duplicate_sequences: bool = True,
    allowed_labels: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    """
    Load STRSeq allele sequences (from your downloaded parquet) into Polars.

    Expected input schema (from scripts/download_strseq.py):
      - label: locus name (e.g. "D1S1656")
      - sample_id: accession-ish identifier
      - sequence: nucleotide sequence

    Returns a DataFrame containing at least ['label', 'sequence'] (and keeps sample_id).
    """
    p = Path(path)
    df = pl.read_parquet(p)

    required = {"label", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{p} is missing required columns: {sorted(missing)} (got {df.columns})"
        )

    # Normalize + basic hygiene
    df = df.with_columns(
        pl.col("label").cast(pl.Utf8).str.strip_chars(),
        pl.col("sequence").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
    )

    if allowed_labels is not None:
        allowed = list(allowed_labels)
        df = df.filter(pl.col("label").is_in(allowed))

    if min_len is not None:
        df = df.filter(pl.col("sequence").str.len_chars() >= int(min_len))

    if max_len is not None:
        df = df.filter(pl.col("sequence").str.len_chars() <= int(max_len))

    if only_acgt:
        # keep sequences made only of A/C/G/T
        df = df.filter(pl.col("sequence").str.contains(r"^[ACGT]+$"))

    if drop_duplicate_sequences:
        # De-dupe identical sequences *within* a locus label
        df = df.unique(subset=["label", "sequence"], keep="first")

    return df
