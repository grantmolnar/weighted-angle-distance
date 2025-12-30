from __future__ import annotations

from typing import Any, Optional, cast

import polars as pl

from src.experiments.dbscan_defaults import DATA_IMPORTERS


def _require_number(x: Optional[Any], *, what: str) -> int:
    if x is None:
        raise ValueError(f"Expected {what} to be non-null, got None")
    # Polars may return int/float/Decimal depending on ops; coerce through float then int.
    return int(float(x))


def main() -> None:
    for imp in DATA_IMPORTERS:
        df: pl.DataFrame = imp.load()

        cols = set(df.columns)
        seq_col = "sequence" if "sequence" in cols else None
        label_col = "label" if "label" in cols else None

        n_rows = df.height
        n_cols = df.width

        msg = f"{imp.name:<12}  rows={n_rows:<7} cols={n_cols:<3}"

        if label_col is not None:
            # n_unique() returns an Expr -> select -> DataFrame; grab scalar safely
            n_labels_df = df.select(pl.col(label_col).n_unique().alias("n_labels"))
            n_labels_val = n_labels_df.item(0, 0)
            msg += f"  n_labels={_require_number(cast(Optional[Any], n_labels_val), what='n_labels')}"

        if seq_col is not None:
            lens = df.select(pl.col(seq_col).str.len_chars().alias("len"))

            # If empty, skip (avoids None)
            if lens.height == 0:
                msg += "  seq_len(min/med/max)=<empty>"
            else:
                stats = lens.select(
                    pl.col("len").min().alias("min"),
                    pl.col("len").median().alias("med"),
                    pl.col("len").max().alias("max"),
                )
                min_v = _require_number(
                    cast(Optional[Any], stats.item(0, 0)), what="min_len"
                )
                med_v = _require_number(
                    cast(Optional[Any], stats.item(0, 1)), what="med_len"
                )
                max_v = _require_number(
                    cast(Optional[Any], stats.item(0, 2)), what="max_len"
                )
                msg += f"  seq_len(min/med/max)={min_v}/{med_v}/{max_v}"

        print(msg)


if __name__ == "__main__":
    main()
