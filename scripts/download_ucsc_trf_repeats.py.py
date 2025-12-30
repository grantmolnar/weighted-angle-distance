from __future__ import annotations

import argparse
import csv
import gzip
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import polars as pl


def _http_get(url: str, *, retries: int = 5, base_sleep: float = 0.5) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(base_sleep * (2**attempt))
    assert last_exc is not None
    raise last_exc


@dataclass(frozen=True)
class TrfRow:
    chrom: str
    start: int
    end: int
    period: int
    copies: float
    unit: str  # repeat unit / "S"


def _iter_trf_bed_gz(path: Path) -> Iterator[TrfRow]:
    """
    UCSC {assembly}.trf.bed.gz is a BED-like, tab-delimited file.
    Observed columns (16 total):
      0 chrom
      1 chromStart
      2 chromEnd
      3 name (usually "trf")
      4 period
      5 copies
      6 consensusSize
      7 perMatch
      8 perIndel
      9 score
      10 A
      11 C
      12 G
      13 T
      14 entropy
      15 sequence (repeat unit)
    """
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if not parts or len(parts) < 16:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            period = int(parts[4])
            copies = float(parts[5])
            unit = parts[15].strip().upper()
            yield TrfRow(chrom=chrom, start=start, end=end, period=period, copies=copies, unit=unit)


def _is_acgt(s: str) -> bool:
    return all(c in {"A", "C", "G", "T"} for c in s)


def _canonical_repeat(unit: str, total_len: int) -> str:
    """
    Produce a canonical "repeat string" of exact length total_len by repeating unit and truncating.
    This intentionally creates the S^m / S^n structure.
    """
    if total_len <= 0:
        return ""
    if not unit:
        return ""
    reps = int(math.ceil(total_len / len(unit)))
    return (unit * reps)[:total_len]


def download_ucsc_trf_repeats(
    *,
    assembly: str,
    out_dir: Path,
    keep_bed_gz: bool,
    min_len: int,
    max_len: int,
    only_acgt: bool,
    top_k_labels: int,
    min_per_label: int,
    max_total_rows: int,
    parquet_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # UCSC publishes this in bigZips: {assembly}.trf.bed.gz :contentReference[oaicite:1]{index=1}
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{assembly}/bigZips/{assembly}.trf.bed.gz"
    bed_gz_path = out_dir / f"{assembly}.trf.bed.gz"

    if not bed_gz_path.exists():
        print(f"Downloading: {url}")
        bed_gz_path.write_bytes(_http_get(url))
    else:
        print(f"Using cached: {bed_gz_path}")

    # First pass: count labels to optionally take top-K motifs.
    label_counts: dict[str, int] = {}
    for row in _iter_trf_bed_gz(bed_gz_path):
        if only_acgt and (not _is_acgt(row.unit)):
            continue
        span = row.end - row.start
        if span < min_len or span > max_len:
            continue
        label_counts[row.unit] = label_counts.get(row.unit, 0) + 1

    allowed_labels: Optional[set[str]] = None
    if top_k_labels > 0:
        # keep labels that are frequent and have at least min_per_label
        items = [(lab, c) for lab, c in label_counts.items() if c >= min_per_label]
        items.sort(key=lambda x: x[1], reverse=True)
        allowed_labels = {lab for lab, _ in items[:top_k_labels]}
        print(
            f"Keeping top_k_labels={top_k_labels} motifs "
            f"(after min_per_label={min_per_label}): {len(allowed_labels)} labels"
        )

    # Second pass: build rows.
    rows: list[Tuple[str, str, str]] = []
    for r in _iter_trf_bed_gz(bed_gz_path):
        if only_acgt and (not _is_acgt(r.unit)):
            continue
        span = r.end - r.start
        if span < min_len or span > max_len:
            continue
        if allowed_labels is not None and r.unit not in allowed_labels:
            continue

        label = r.unit
        sample_id = f"{assembly}:{r.chrom}:{r.start}-{r.end}"
        seq = _canonical_repeat(r.unit, span)
        if not seq:
            continue

        rows.append((label, sample_id, seq))
        if max_total_rows > 0 and len(rows) >= max_total_rows:
            break

    df = pl.DataFrame(rows, schema=["label", "sample_id", "sequence"])
    df.write_parquet(parquet_path)
    print(f"Wrote parquet: {parquet_path}  (rows={df.height}, labels={df['label'].n_unique()})")

    if not keep_bed_gz:
        bed_gz_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download UCSC TRF tandem repeats and build a labeled parquet (label=repeat unit)."
    )
    p.add_argument("--assembly", type=str, default="hg19", help="Genome assembly (e.g. hg19, hg38).")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/ucsc_trf"))
    p.add_argument("--parquet", type=Path, default=Path("data/processed/ucsc_trf.parquet"))
    p.add_argument("--keep-bed", action="store_true", help="Keep downloaded {assembly}.trf.bed.gz.")
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=600)
    p.add_argument("--only-acgt", action="store_true", help="Filter to motifs over A/C/G/T only.")
    p.add_argument("--top-k-labels", type=int, default=200, help="Keep only the top-K most frequent motifs (0=all).")
    p.add_argument("--min-per-label", type=int, default=25, help="Only consider motifs with >= this many samples.")
    p.add_argument("--max-total-rows", type=int, default=50000, help="Cap output size (0=no cap).")
    args = p.parse_args()

    download_ucsc_trf_repeats(
        assembly=args.assembly,
        out_dir=args.out_dir,
        keep_bed_gz=args.keep_bed,
        min_len=args.min_len,
        max_len=args.max_len,
        only_acgt=args.only_acgt,
        top_k_labels=args.top_k_labels,
        min_per_label=args.min_per_label,
        max_total_rows=args.max_total_rows,
        parquet_path=args.parquet,
    )


if __name__ == "__main__":
    main()

# Example:
# python -m scripts.download_ucsc_trf_repeats --assembly hg19 --only-acgt \
#   --min-len 20 --max-len 600 --top-k-labels 200 --min-per-label 25 --max-total-rows 50000 \
#   --parquet data/processed/ucsc_trf_hg19.parquet
