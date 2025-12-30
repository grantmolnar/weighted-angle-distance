from __future__ import annotations

import argparse
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

# NOTE:
# These URLs have historically worked, but TalkBank occasionally reorganizes paths.
# If a download fails, open the corpus page and copy the "Download transcripts" link,
# then pass it via --zip-url.
DEFAULT_CORPUS_URLS = {
    "hakim": "https://git.talkbank.org/fluency/data/Hakim.zip",
    # optional (often gated / "Password" directory; leave here if you have access)
    "iisrp": "https://git.talkbank.org/fluency/data/Password/IISRP.zip",
}


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


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = _http_get(url)
    dst.write_bytes(data)


def _extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # keep only .cha files (CHAT transcripts)
            if not info.filename.lower().endswith(".cha"):
                continue
            zf.extract(info, out_dir)
            extracted.append(out_dir / info.filename)
    return extracted


def _read_text_best_effort(path: Path) -> str:
    # TalkBank CHAT files are usually UTF-8, but be defensive.
    b = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def _iter_chat_main_tiers(chat_text: str) -> Iterator[Tuple[str, str]]:
    """
    Yield (speaker_code, utterance_text) for main tiers in a CHAT .cha file.

    Main tiers look like:
        *CHI:  ...utterance...
        *MOT:  ...utterance...
    Continuation lines often begin with a tab.
    """
    cur_speaker: Optional[str] = None
    cur_utt_parts: List[str] = []

    for raw in chat_text.splitlines():
        line = raw.rstrip("\n")

        if line.startswith("*") and ":" in line[:8]:
            # flush previous
            if cur_speaker is not None:
                yield cur_speaker, " ".join(cur_utt_parts).strip()
            cur_utt_parts = []

            # parse "*XXX: rest"
            speaker = line[1: line.index(":")].strip()
            rest = line[line.index(":") + 1 :].strip()
            cur_speaker = speaker
            if rest:
                cur_utt_parts.append(rest)
            continue

        # continuation of prior main tier
        if cur_speaker is not None and (line.startswith("\t") or (line and not line[0].isalpha() and not line.startswith(("@", "%", "*")))):
            cont = line.strip()
            if cont:
                cur_utt_parts.append(cont)
            continue

        # ignore headers (@), dependent tiers (%), blank lines, etc.
        continue

    if cur_speaker is not None:
        yield cur_speaker, " ".join(cur_utt_parts).strip()


_FILLED_PAUSES = {"um", "uh", "erm", "er", "mm", "mhm", "hmm"}


def encode_utterance_to_pattern(utt: str) -> str:
    """
    Convert a CHAT utterance into a compact string over a small alphabet.

    Intended to make string distances compare "fluency structure" rather than topic/vocabulary.

    Alphabet (default):
      W = (mostly) lexical token
      F = filled pause (um/uh/...)
      R = explicit repetition marker token (e.g., [/] or [//])
      P = prolongation-ish token (heuristic: contains '::' or many ':')
      S = pause token like (.) (..) (...)
      X = unintelligible marker (xxx, yyy)

    This is heuristic but works surprisingly well for clustering demos.
    """
    # normalize spacing
    s = re.sub(r"\s+", " ", utt.strip())
    if not s:
        return ""

    toks = s.split(" ")
    out: List[str] = []

    for t in toks:
        tl = t.lower()

        # common CHAT markers
        if tl in {"xxx", "yyy"}:
            out.append("X")
            continue
        if tl in {"[/]", "[//]"}:
            out.append("R")
            continue
        if re.fullmatch(r"\(\.+\)", tl):  # (.) (..) (...)
            out.append("S")
            continue

        # filled pauses (raw or with CHAT prefixes)
        tl_stripped = re.sub(r"^[&][+-]?", "", tl)  # &+uh, &-um, etc.
        if tl_stripped in _FILLED_PAUSES:
            out.append("F")
            continue

        # prolongation-ish: "so::" or "s:::o"
        if "::" in t or t.count(":") >= 2:
            out.append("P")
            continue

        # strip obvious punctuation but keep brackets/parens markers handled above
        core = re.sub(r"[^\w'-]+", "", t)
        if not core:
            continue
        out.append("W")

    return "".join(out)


def encode_utterance_raw_chars(utt: str) -> str:
    """
    Alternative: keep a character-level string (uppercase letters + a few markers),
    removing spaces. Useful if you want a "closer to raw transcript" baseline.
    """
    s = utt.strip().upper()
    if not s:
        return ""
    # keep A-Z plus a handful of CHAT punctuation markers that might reflect stutter structure
    s = re.sub(r"[^A-Z\[\]/():.&+-]+", "", s)
    return s


def infer_label(
    *,
    corpus: str,
    chat_text: str,
    rel_path: str,
    label_mode: str,
) -> str:
    """
    label_mode:
      - "corpus": label = corpus name (e.g., "hakim")
      - "group": attempt to infer CWS vs CONTROL (works well for Hakim if directories/headers include it)
      - "dysfluent": label utterance-level later; here we just return corpus (unused)
    """
    if label_mode == "corpus":
        return corpus

    if label_mode == "group":
        # 1) try path hints
        up = rel_path.upper()
        if "CWS" in up or "STUT" in up:
            return "CWS"
        if "CONTROL" in up or "CTL" in up or "TYP" in up:
            return "CONTROL"

        # 2) try header hints
        # Look for common TalkBank @ID or @Participants fields that might mention group.
        header_chunk = "\n".join(chat_text.splitlines()[:120]).upper()
        if "CWS" in header_chunk or "STUT" in header_chunk:
            return "CWS"
        if "CONTROL" in header_chunk or "TYPICAL" in header_chunk:
            return "CONTROL"

        # 3) fallback: corpus label
        return corpus

    if label_mode == "dysfluent":
        return corpus

    raise ValueError(f"Unknown label_mode={label_mode!r}")


@dataclass(frozen=True)
class Row:
    label: str
    sample_id: str
    sequence: str


def build_rows_from_cha_files(
    *,
    corpus: str,
    cha_paths: Iterable[Path],
    root_dir: Path,
    label_mode: str,
    encoding: str,
    min_len: int,
    max_len: int,
    speakers: Optional[set[str]],
) -> List[Row]:
    rows: List[Row] = []

    for p in cha_paths:
        rel = str(p.relative_to(root_dir)).replace("\\", "/")
        txt = _read_text_best_effort(p)

        file_label = infer_label(
            corpus=corpus, chat_text=txt, rel_path=rel, label_mode=label_mode
        )

        utt_idx = 0
        for spk, utt in _iter_chat_main_tiers(txt):
            if speakers is not None and spk not in speakers:
                continue

            if encoding == "pattern":
                seq = encode_utterance_to_pattern(utt)
            elif encoding == "raw_chars":
                seq = encode_utterance_raw_chars(utt)
            else:
                raise ValueError(f"Unknown encoding={encoding!r}")

            if not seq:
                continue
            if len(seq) < min_len:
                continue
            if max_len > 0 and len(seq) > max_len:
                continue

            sample_id = f"{rel}::spk={spk}::utt={utt_idx}"
            rows.append(Row(label=file_label, sample_id=sample_id, sequence=seq))
            utt_idx += 1

    # If label_mode is dysfluent, relabel based on sequence content (simple example):
    if label_mode == "dysfluent":
        # For pattern encoding, treat presence of R/P/F/S as dysfluent.
        # (Tune as desired; you can also do multi-class by "dominant marker".)
        new_rows: List[Row] = []
        for r in rows:
            if any(c in r.sequence for c in ("R", "P", "F", "S")):
                lab = "DYSFLUENT"
            else:
                lab = "FLUENT"
            new_rows.append(Row(label=lab, sample_id=r.sample_id, sequence=r.sequence))
        rows = new_rows

    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download + process a FluencyBank corpus into label/sample_id/sequence parquet."
    )
    p.add_argument(
        "--corpus",
        type=str,
        default="hakim",
        choices=sorted(DEFAULT_CORPUS_URLS.keys()),
        help="Which FluencyBank corpus to download.",
    )
    p.add_argument(
        "--zip-url",
        type=str,
        default="",
        help="Override the transcript ZIP URL (if TalkBank paths change or access differs).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/fluencybank"),
        help="Where to store downloaded zip + extracted .cha files.",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/fluencybank_hakim_utterances.parquet"),
        help="Output parquet path.",
    )
    p.add_argument(
        "--label-mode",
        type=str,
        default="group",
        choices=["group", "corpus", "dysfluent"],
        help="How to assign ground-truth labels.",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="pattern",
        choices=["pattern", "raw_chars"],
        help="How to turn an utterance into a string.",
    )
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=400, help="0 means no max.")
    p.add_argument(
        "--speakers",
        type=str,
        default="",
        help="Comma-separated speaker codes to keep (e.g. CHI). Empty means keep all.",
    )
    args = p.parse_args()

    url = args.zip_url.strip() or DEFAULT_CORPUS_URLS[args.corpus]

    corpus_dir = args.raw_dir / args.corpus
    zip_path = corpus_dir / f"{args.corpus}.zip"
    extracted_dir = corpus_dir / "cha"

    print(f"Downloading: {url}")
    _download_file(url, zip_path)
    print(f"Wrote: {zip_path}")

    cha_paths = _extract_zip(zip_path, extracted_dir)
    print(f"Extracted .cha files: {len(cha_paths)} -> {extracted_dir}")

    speakers = None
    if args.speakers.strip():
        speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}

    rows = build_rows_from_cha_files(
        corpus=args.corpus,
        cha_paths=cha_paths,
        root_dir=extracted_dir,
        label_mode=args.label_mode,
        encoding=args.encoding,
        min_len=args.min_len,
        max_len=args.max_len,
        speakers=speakers,
    )

    import polars as pl

    df = pl.DataFrame(
        [(r.label, r.sample_id, r.sequence) for r in rows],
        schema=["label", "sample_id", "sequence"],
    )
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.parquet)
    print(f"Wrote parquet: {args.parquet}  (rows={df.height})")


if __name__ == "__main__":
    main()

# Example:
# python -m scripts.download_fluencybank_data \
#   --corpus hakim \
#   --raw-dir data/raw/fluencybank \
#   --parquet data/processed/fluencybank_hakim_utterances.parquet \
#   --label-mode group \
#   --encoding pattern \
#   --speakers CHI
