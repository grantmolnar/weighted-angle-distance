# scripts/make_synthetic_tr.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.synthetic_tandem_repeats import (
    DEFAULT_SYNTHETIC_TR_CONFIG,
    SyntheticTandemRepeatConfig,
    ensure_synthetic_tandem_repeat_dataset,
)


def _repo_root() -> Path:
    # repo_root/scripts/make_synthetic_tr.py -> repo_root
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a synthetic tandem-repeat dataset and write it to parquet.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "src" / "data" / "synthetic_tandem_repeats.parquet",
        help="Output parquet path.",
    )
    p.add_argument(
        "--force", action="store_true", help="Overwrite output if it already exists."
    )
    p.add_argument(
        "--seed-labels", type=int, default=0, help="Seed for generating class labels."
    )
    p.add_argument(
        "--seed-samples", type=int, default=1, help="Seed for per-sample randomness."
    )

    # Dataset size / class structure
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.n_samples
    )
    p.add_argument(
        "--n-classes", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.n_classes
    )
    p.add_argument(
        "--motifs-per-label-min",
        type=int,
        default=DEFAULT_SYNTHETIC_TR_CONFIG.motifs_per_label_min,
    )
    p.add_argument(
        "--motifs-per-label-max",
        type=int,
        default=DEFAULT_SYNTHETIC_TR_CONFIG.motifs_per_label_max,
    )

    # Alphabet and motif lengths
    p.add_argument("--alphabet", type=str, default=DEFAULT_SYNTHETIC_TR_CONFIG.alphabet)
    p.add_argument(
        "--motif-len-min", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.motif_len_min
    )
    p.add_argument(
        "--motif-len-max", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.motif_len_max
    )

    # Flanks and separators
    p.add_argument(
        "--flank-len-min", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.flank_len_min
    )
    p.add_argument(
        "--flank-len-max", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.flank_len_max
    )
    p.add_argument(
        "--sep-len-min", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.sep_len_min
    )
    p.add_argument(
        "--sep-len-max", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.sep_len_max
    )

    # Length normalization / repeats
    p.add_argument(
        "--target-expected-total-length",
        type=float,
        default=DEFAULT_SYNTHETIC_TR_CONFIG.target_expected_total_length,
        help="Controls the approximate expected overall sample length.",
    )
    p.add_argument(
        "--repeat-cap", type=int, default=DEFAULT_SYNTHETIC_TR_CONFIG.repeat_cap
    )

    # Noise
    p.add_argument(
        "--mutation-rate-non-motif",
        type=float,
        default=DEFAULT_SYNTHETIC_TR_CONFIG.mutation_rate_non_motif,
    )

    # Label formatting
    p.add_argument(
        "--label-sep", type=str, default=DEFAULT_SYNTHETIC_TR_CONFIG.label_sep
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    out_path: Path = args.out
    if args.force and out_path.exists():
        out_path.unlink()

    cfg = SyntheticTandemRepeatConfig(
        n_samples=int(args.n_samples),
        n_classes=int(args.n_classes),
        motifs_per_label_min=int(args.motifs_per_label_min),
        motifs_per_label_max=int(args.motifs_per_label_max),
        alphabet=str(args.alphabet),
        motif_len_min=int(args.motif_len_min),
        motif_len_max=int(args.motif_len_max),
        flank_len_min=int(args.flank_len_min),
        flank_len_max=int(args.flank_len_max),
        sep_len_min=int(args.sep_len_min),
        sep_len_max=int(args.sep_len_max),
        target_expected_total_length=float(args.target_expected_total_length),
        repeat_cap=int(args.repeat_cap),
        mutation_rate_non_motif=float(args.mutation_rate_non_motif),
        label_sep=str(args.label_sep),
    )

    df = ensure_synthetic_tandem_repeat_dataset(
        out_path,
        cfg,
        seed_labels=int(args.seed_labels),
        seed_samples=int(args.seed_samples),
    )

    print("Wrote / loaded synthetic dataset:")
    print(f"  path: {out_path}")
    print(f"  rows: {df.height}")
    print(f"  cols: {df.columns}")
    print("Config:")
    print(f"  n_samples={cfg.n_samples}, n_classes={cfg.n_classes}")
    print(
        "  motifs_per_label=[%d,%d], motif_len=[%d,%d]"
        % (
            cfg.motifs_per_label_min,
            cfg.motifs_per_label_max,
            cfg.motif_len_min,
            cfg.motif_len_max,
        )
    )
    print(
        "  flank_len=[%d,%d], sep_len=[%d,%d]"
        % (cfg.flank_len_min, cfg.flank_len_max, cfg.sep_len_min, cfg.sep_len_max)
    )
    print(
        "  target_expected_total_length=%.3f, repeat_cap=%d, mutation_rate_non_motif=%.6f"
        % (
            cfg.target_expected_total_length,
            cfg.repeat_cap,
            cfg.mutation_rate_non_motif,
        )
    )
    print(f"  label_sep={cfg.label_sep!r}")
    print(
        f"Seeds: seed_labels={int(args.seed_labels)}, seed_samples={int(args.seed_samples)}"
    )


if __name__ == "__main__":
    main()
