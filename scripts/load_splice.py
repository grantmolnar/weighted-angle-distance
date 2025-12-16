from __future__ import annotations

import argparse
from pathlib import Path

from src.data.splice_loader import load_splice_to_polars


def main() -> None:
    parser = argparse.ArgumentParser(description="Load splice.data into Polars.")
    parser.add_argument("path", type=Path, help="Path to splice.data")
    parser.add_argument("--out", type=Path, default=None, help="Optional output parquet path")
    args = parser.parse_args()

    df = load_splice_to_polars(args.path)
    print(df.shape)
    print(df.head())

    if args.out is not None:
        df.write_parquet(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
