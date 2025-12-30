#!/usr/bin/env python3
"""
download_splice.py

Download and normalize the UCI "Molecular Biology (Splice-junction Gene Sequences)"
dataset into a 3-column CSV compatible with:

    load_splice_to_polars(path)  # expects rows: label,sample_id,sequence

Output row format:
    label,sample_id,sequence

Where:
  - label is one of: EI, IE, N
  - sample_id is the instance name/id from the source file
  - sequence is a 60-character DNA-like string (A,C,G,T plus ambiguity codes)

Typical usage:
  python download_splice.py --out data/splice.data
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# UCI provides the dataset as a zip. Keep a few mirrors/candidates for robustness.
CANDIDATE_URLS: list[str] = [
    # Current UCI CDN (dataset id 69)
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences.zip",
    # UCI static mirror (often works if CDN changes)
    "https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip",
    "https://archive.ics.uci.edu/static/public/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences.zip",
    # Legacy locations (may or may not still be available)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/splice/splice.data",
]


def _http_get_bytes(url: str, *, timeout_s: int = 60) -> bytes:
    # Set a UA header; some hosts block the default Python UA.
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (download_splice.py)"})
    with urlopen(req, timeout=timeout_s) as r:
        return r.read()


def _download_from_candidates(urls: Iterable[str]) -> tuple[str, bytes]:
    last_err: Exception | None = None
    for url in urls:
        try:
            data = _http_get_bytes(url)
            if not data:
                raise RuntimeError("Empty response body.")
            return url, data
        except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to download from all candidate URLs. Last error: {last_err}"
    )


def _extract_splice_payload(downloaded: bytes) -> tuple[str, bytes]:
    """
    Returns (source_name, payload_bytes)

    Supports:
      - raw text splice.data (already CSV-ish)
      - zip containing splice.data (and possibly other files)
    """
    # ZIP magic header: b"PK\x03\x04"
    if downloaded[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(downloaded)) as zf:
            names = zf.namelist()

            # Prefer a file explicitly named splice.data (any folder depth).
            def score(n: str) -> tuple[int, int]:
                ln = n.lower()
                return (
                    0 if ln.endswith("splice.data") else 1,
                    0 if ln.endswith(".data") else 1,
                )

            candidates = sorted(names, key=score)
            pick = None
            for n in candidates:
                ln = n.lower()
                if ln.endswith("splice.data") or ln.endswith(".data"):
                    pick = n
                    break
            if pick is None:
                raise RuntimeError(
                    f"Zip did not contain a .data file. Contents: {names[:50]}"
                )

            return pick, zf.read(pick)

    # Otherwise treat as a raw text file
    return "splice.data", downloaded


def _normalize_label(raw: str) -> str:
    lab = raw.strip().upper()
    # Common variants: "EI", "IE", "N" (sometimes "NEITHER" or lowercase)
    if lab in {"EI", "IE", "N"}:
        return lab
    if lab in {"NEITHER", "NONE"}:
        return "N"
    # Fall back: keep uppercased label (caller can inspect)
    return lab


def _build_sequence(fields_after_id: list[str]) -> str:
    """
    Build a 60-char sequence from either:
      - a single string field that may contain spaces between letters
      - 60 separate fields (one char each)
    """
    # Join then remove whitespace characters.
    joined = "".join(s.strip() for s in fields_after_id)
    joined = "".join(ch for ch in joined if not ch.isspace())
    return joined.upper()


def normalize_splice_rows(raw_text: str) -> list[tuple[str, str, str]]:
    """
    Parse the source splice.data into (label, sample_id, sequence) rows.

    Handles two common source formats:
      (A) 3 columns: label, id, "sequence" (possibly with spaces)
      (B) 62 columns: label, id, 60 nucleotide fields
    """
    out: list[tuple[str, str, str]] = []
    reader = csv.reader(io.StringIO(raw_text))
    for row_idx, row in enumerate(reader, start=1):
        if not row:
            continue
        fields = [c.strip() for c in row]
        if len(fields) < 3:
            # Unexpected line; skip quietly but leave a breadcrumb on stderr.
            print(
                f"[splice] skipping row {row_idx}: expected >=3 columns, got {len(fields)}",
                file=sys.stderr,
            )
            continue

        label = _normalize_label(fields[0])
        sample_id = fields[1].strip()

        if len(fields) == 3:
            # Third field might be "a g g t ..." or "aggtt..."
            seq = _build_sequence([fields[2]])
        else:
            # Remaining fields represent positions; join them.
            seq = _build_sequence(fields[2:])

        # Optional sanity checks (do not hard-fail; the dataset has rare ambiguity codes).
        if len(seq) != 60:
            print(
                f"[splice] warning row {row_idx}: sequence length {len(seq)} != 60 (label={label}, id={sample_id})",
                file=sys.stderr,
            )

        out.append((label, sample_id, seq))
    return out


def write_three_column_csv(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download and normalize the UCI splice-junction dataset."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/splice.data"),
        help="Where to write the cleaned, loader-ready splice.data (default: data/processed/splice.data).",
    )
    ap.add_argument(
        "--url",
        type=str,
        default="",
        help="Optional: explicit URL to download from (overrides built-in candidates).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    ap.add_argument(
        "--also-write-raw",
        type=Path,
        default=Path("data/raw/splice.data"),
        help="Optional: also write the raw extracted splice.data here (default: data/raw/splice.data).",
    )

    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(
            f"[splice] Refusing to overwrite existing file: {args.out} (use --force)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    urls = [args.url] if args.url else CANDIDATE_URLS
    used_url, downloaded = _download_from_candidates(urls)
    source_name, payload = _extract_splice_payload(downloaded)

    # Decode as text; tolerate odd encodings via replacement.
    text = payload.decode("utf-8", errors="replace")

    if args.also_write_raw is not None:
        args.also_write_raw.parent.mkdir(parents=True, exist_ok=True)
        args.also_write_raw.write_text(text, encoding="utf-8")

    rows = normalize_splice_rows(text)
    write_three_column_csv(rows, args.out)

    print(f"[splice] downloaded from: {used_url}")
    print(f"[splice] extracted: {source_name}")
    print(f"[splice] wrote: {args.out}  (rows={len(rows)})")


if __name__ == "__main__":
    main()
