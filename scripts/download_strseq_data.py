from __future__ import annotations

import argparse
import gzip
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# --- PRJNA -> locus label (your 24 subprojects) ---
STRSEQ_PROJECTS: Dict[str, str] = {
    "PRJNA380553": "D1S1656",
    "PRJNA380554": "TPOX",
    "PRJNA380555": "D2S441",
    "PRJNA380556": "D2S1338",
    "PRJNA380558": "D3S1358",
    "PRJNA380559": "FGA",
    "PRJNA380560": "D5S818",
    "PRJNA380561": "CSF1PO",
    "PRJNA380562": "SE33",
    "PRJNA380563": "D6S1043",
    "PRJNA380564": "D7S820",
    "PRJNA380565": "D8S1179",
    "PRJNA380566": "D10S1248",
    "PRJNA380567": "TH01",
    "PRJNA380568": "vWA",
    "PRJNA380569": "D12S391",
    "PRJNA380570": "D13S317",
    "PRJNA380571": "PentaE",
    "PRJNA380572": "D16S539",
    "PRJNA380573": "D18S51",
    "PRJNA380574": "D19S433",
    "PRJNA380575": "D21S11",
    "PRJNA380576": "PentaD",
    "PRJNA380577": "D22S1045",
}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _http_get(url: str, *, retries: int = 5, base_sleep: float = 0.5) -> bytes:
    """
    Simple GET with exponential backoff retries (for transient network/NCBI hiccups).
    """
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


def _esearch_usehistory(
    *,
    db: str,
    term: str,
    email: str,
    tool: str,
    api_key: Optional[str],
) -> Tuple[int, str, str]:
    """
    Run ESearch with usehistory=y and return (count, query_key, webenv).
    """
    params = {
        "db": db,
        "term": term,
        "usehistory": "y",
        "retmode": "xml",
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    xml_bytes = _http_get(url)

    root = ET.fromstring(xml_bytes)
    count = int(root.findtext("Count", default="0"))
    query_key = root.findtext("QueryKey", default="")
    webenv = root.findtext("WebEnv", default="")
    return count, query_key, webenv


def _efetch_fasta_chunks(
    *,
    db: str,
    query_key: str,
    webenv: str,
    total: int,
    email: str,
    tool: str,
    api_key: Optional[str],
    retmax: int,
    sleep_seconds: float,
) -> Iterator[str]:
    """
    Yield FASTA text in chunks using EFetch + history (WebEnv/QueryKey).
    """
    for retstart in range(0, total, retmax):
        params = {
            "db": db,
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": str(retstart),
            "retmax": str(retmax),
            "rettype": "fasta",
            "retmode": "text",
            "tool": tool,
            "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        url = f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        text = _http_get(url).decode("utf-8", errors="replace")
        yield text
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _parse_fasta_records(fasta_text: str) -> Iterator[Tuple[str, str]]:
    """
    Parse FASTA into (header, sequence). header excludes the leading '>'.
    """
    header: Optional[str] = None
    seq_parts: List[str] = []

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)

    if header is not None:
        yield header, "".join(seq_parts)


def _header_to_accession(header: str) -> str:
    """
    Best-effort extraction of an accession-like id from a FASTA header.
    """
    first = header.split()[0]
    if "|" in first:
        # e.g. gi|...|gb|ABC123.1|  -> ABC123.1
        parts = [p for p in first.split("|") if p]
        return parts[-1] if parts else first
    return first


def download_one_bioproject(
    *,
    bioproject: str,
    label: str,
    out_dir: Path,
    email: str,
    tool: str,
    api_key: Optional[str],
    gzip_fasta: bool,
    retmax: int,
    sleep_seconds: float,
) -> Tuple[Path, List[Tuple[str, str, str]]]:
    """
    Download nuccore FASTA records for one BioProject and return:
      (fasta_path, rows_for_parquet=[(label, sample_id, sequence), ...])
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    term = f"{bioproject}[BioProject]"
    total, qk, we = _esearch_usehistory(
        db="nuccore", term=term, email=email, tool=tool, api_key=api_key
    )

    suffix = ".fasta.gz" if gzip_fasta else ".fasta"
    fasta_path = out_dir / f"{bioproject}__{label}{suffix}"

    rows: List[Tuple[str, str, str]] = []

    if total == 0:
        # Create empty file so downstream scripts can see it existed.
        if gzip_fasta:
            with gzip.open(fasta_path, "wt", encoding="utf-8") as f:
                f.write("")
        else:
            fasta_path.write_text("", encoding="utf-8")
        return fasta_path, rows

    if gzip_fasta:
        with gzip.open(fasta_path, "wt", encoding="utf-8", newline="\n") as f:
            for chunk in _efetch_fasta_chunks(
                db="nuccore",
                query_key=qk,
                webenv=we,
                total=total,
                email=email,
                tool=tool,
                api_key=api_key,
                retmax=retmax,
                sleep_seconds=sleep_seconds,
            ):
                f.write(chunk)
                for header, seq in _parse_fasta_records(chunk):
                    sample_id = _header_to_accession(header)
                    rows.append((label, sample_id, seq))
    else:
        with open(fasta_path, "w", encoding="utf-8", newline="\n") as f:
            for chunk in _efetch_fasta_chunks(
                db="nuccore",
                query_key=qk,
                webenv=we,
                total=total,
                email=email,
                tool=tool,
                api_key=api_key,
                retmax=retmax,
                sleep_seconds=sleep_seconds,
            ):
                f.write(chunk)
                for header, seq in _parse_fasta_records(chunk):
                    sample_id = _header_to_accession(header)
                    rows.append((label, sample_id, seq))

    return fasta_path, rows



def main() -> None:
    p = argparse.ArgumentParser(
        description="Download STRSeq allele sequences (nuccore FASTA) for STRSeq BioProjects."
    )
    p.add_argument(
        "--projects",
        nargs="*",
        default=[],
        help="BioProject accessions to download (e.g. PRJNA380553). Default: all known STRSeq subprojects.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("src/data/strseq_fasta"))
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional: write combined parquet with label/sample_id/sequence.",
    )
    p.add_argument(
        "--email",
        type=str,
        default=os.environ.get("NCBI_EMAIL", ""),
        help="NCBI contact email (or set NCBI_EMAIL).",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NCBI_API_KEY", ""),
        help="NCBI API key (or set NCBI_API_KEY).",
    )
    p.add_argument("--tool", type=str, default="weighted-angle-distance")
    p.add_argument("--gzip", action="store_true", help="Write gzipped FASTA files.")
    p.add_argument("--retmax", type=int, default=500, help="EFetch chunk size.")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.34,
        help="Seconds to sleep between requests (keep ~0.34 for ~3 req/s without API key).",
    )
    args = p.parse_args()

    if not args.email:
        raise SystemExit(
            "ERROR: please pass --email or set NCBI_EMAIL (NCBI requests you identify yourself)."
        )

    api_key = args.api_key or None

    projects = args.projects or list(STRSEQ_PROJECTS.keys())
    unknown = [p for p in projects if p not in STRSEQ_PROJECTS]
    if unknown:
        raise SystemExit(
            f"Unknown projects (not in STRSEQ_PROJECTS mapping): {unknown}"
        )

    all_rows: List[Tuple[str, str, str]] = []

    for bp in projects:
        label = STRSEQ_PROJECTS[bp]
        fasta_path, rows = download_one_bioproject(
            bioproject=bp,
            label=label,
            out_dir=args.out_dir,
            email=args.email,
            tool=args.tool,
            api_key=api_key,
            gzip_fasta=args.gzip,
            retmax=args.retmax,
            sleep_seconds=args.sleep,
        )
        print(f"{bp} ({label}): wrote {fasta_path} with {len(rows)} sequences")
        all_rows.extend(rows)

    if args.parquet is not None:
        import polars as pl

        df = pl.DataFrame(all_rows, schema=["label", "sample_id", "sequence"])
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.parquet)
        print(f"Wrote parquet: {args.parquet}  (rows={df.height})")


if __name__ == "__main__":
    main()

# python -m scripts.download_strseq_data --out-dir data/raw/strseq_fasta --parquet data/processed/strseq_alleles.parquet --gzip --email myemail@gmail.com
