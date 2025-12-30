from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import polars as pl


def load_splice_to_polars(path: Union[str, Path]) -> pl.DataFrame:
    """
    Load the UCI Splice Junction 'splice.data' file into a Polars DataFrame.

    The file is comma-separated with three fields per row:
      1) label (e.g., 'EI', 'IE', 'N')
      2) sample_id (an identifier string)
      3) sequence (a fixed-length DNA-like string)

    Many versions of the file include extra whitespace around fields, so we
    strip whitespace from each field as we read it.

    Parameters
    ----------
    path:
        Path to the 'splice.data' file. May be a string or pathlib.Path.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with columns: ["label", "sample_id", "sequence"].
    """
    p = Path(path)

    rows: list[tuple[str, str, str]] = []
    with p.open(newline="") as f:
        reader = csv.reader(f)
        for label, sample_id, seq in reader:
            # Strip whitespace around each field (common in this dataset).
            rows.append((label.strip(), sample_id.strip(), seq.strip()))

    # Construct the Polars DataFrame with an explicit schema (stable column order/types).
    return pl.DataFrame(rows, schema=["label", "sample_id", "sequence"], orient="row")
