# src/data/synthetic_tandem_repeats.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SyntheticTandemRepeatConfig:
    """
    Synthetic tandem-repeat dataset generator configuration.

    Summary
    -------
    - Alphabet: DNA-like (default: A,C,G,T)
    - There are `n_classes` classes; each class label is a tuple of `k` motifs
      (k in [motifs_per_label_min, motifs_per_label_max]).
    - Each motif is a short string of length in [motif_len_min, motif_len_max].
    - A sample sequence is:
        prefix + motif_1^r1 + sep_1 + motif_2^r2 + ... + sep_{k-1} + motif_k^rk + suffix
      where prefix/suffix and separators are random DNA strings of lengths in
      [flank_len_min, flank_len_max] and [sep_len_min, sep_len_max].
    - Repeat counts are sampled so that the *expected* total sample length is
      approximately independent of k and motif lengths (up to truncation/capping).
    """

    # Dataset size / class structure
    n_samples: int
    n_classes: int
    motifs_per_label_min: int
    motifs_per_label_max: int

    # Alphabet and motif lengths
    alphabet: str
    motif_len_min: int
    motif_len_max: int

    # Flanks and separators
    flank_len_min: int
    flank_len_max: int
    sep_len_min: int
    sep_len_max: int

    # Length-normalization target (controls expected overall sample length)
    target_expected_total_length: float

    # control whether we normalize lengths (True) or allow wild length variation (False)
    coerce_equal_length: bool

    # Repeat-count distribution and caps
    repeat_cap: int  # hard cap on repeats for any single motif (keeps tails bounded)

    # Optional point-mutation noise (applied to non-motif regions only)
    mutation_rate_non_motif: float  # e.g. 0.0, 0.005, 0.01

    # Label formatting
    label_sep: str  # how motif-tuples are rendered into a single string label


DEFAULT_SYNTHETIC_TR_CONFIG = SyntheticTandemRepeatConfig(
    # Requested: 10,000 samples from 10 classes (balanced generation)
    n_samples=10_000,
    n_classes=10,
    motifs_per_label_min=1,
    motifs_per_label_max=6,
    # Requested: motifs of length 1..8 over DNA alphabet
    alphabet="ACGT",
    motif_len_min=1,
    motif_len_max=8,
    # Requested: separators and prefixes/suffixes of length 0..8
    flank_len_min=0,
    flank_len_max=8,
    sep_len_min=0,
    sep_len_max=0,
    # Choose a single “budget” so expected length is stable across classes.
    # Must be large enough that even (k=6, motif_len=8) can have mean repeats >= 1.
    target_expected_total_length=100.0,
    # If True, we are obliged to keep our strings close to the same length
    coerce_equal_length=False,
    # Keeps very short motifs from occasionally producing very long strings
    repeat_cap=120,
    # Default: no point mutation (easy to turn on later)
    mutation_rate_non_motif=0.0,
    label_sep="|",
)


def _rand_int_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi]."""
    if hi < lo:
        raise ValueError(f"invalid integer range [{lo},{hi}]")
    return int(rng.integers(lo, hi + 1))


def _random_string(rng: np.random.Generator, alphabet: str, length: int) -> str:
    """Uniform random string of fixed length from `alphabet`."""
    if length <= 0:
        return ""
    idx = rng.integers(0, len(alphabet), size=length)
    return "".join(alphabet[i] for i in idx.tolist())


def _mutate_string_nonmotif(
    rng: np.random.Generator,
    s: str,
    alphabet: str,
    mutation_rate: float,
) -> str:
    """
    Independently mutate each character with probability `mutation_rate`
    (substitute with a different character from alphabet).
    """
    if mutation_rate <= 0.0 or not s:
        return s

    out = list(s)
    for i, ch in enumerate(out):
        if rng.random() < mutation_rate:
            # pick a different symbol
            choices = [a for a in alphabet if a != ch]
            out[i] = choices[_rand_int_inclusive(rng, 0, len(choices) - 1)]
    return "".join(out)


def _expected_uniform_int(lo: int, hi: int) -> float:
    """E[U] for U ~ Uniform{lo, lo+1, ..., hi}."""
    return 0.5 * (lo + hi)


def _mean_repeats_for_class_and_motif(
    cfg: SyntheticTandemRepeatConfig,
    *,
    k: int,
    motif_len: int,
) -> float:
    """
    Choose mean repeats mu so that expected total length is (approximately) constant.

    Let:
      L_total_target = cfg.target_expected_total_length
      E_flanks = E[prefix] + E[suffix]
      E_seps = (k-1) * E[separator]

    We allocate the remaining expected length equally among the k motif blocks,
    then divide by motif_len to get mean repeats.

    mu = ((L_total_target - E_flanks - E_seps) / k) / motif_len
    """
    e_flank = 2.0 * _expected_uniform_int(cfg.flank_len_min, cfg.flank_len_max)
    e_sep = _expected_uniform_int(cfg.sep_len_min, cfg.sep_len_max)
    e_seps = float(max(k - 1, 0)) * e_sep

    motif_budget_total = float(cfg.target_expected_total_length) - e_flank - e_seps
    if motif_budget_total <= 0:
        raise ValueError(
            "target_expected_total_length too small relative to expected flanks/separators"
        )

    mu = (motif_budget_total / float(k)) / float(motif_len)
    if mu < 1.0:
        # Support of repeat count is {1,2,...}, so mean < 1 is impossible.
        # This is a configuration error (increase target_expected_total_length).
        raise ValueError(
            f"Configuration yields mean repeats < 1 (mu={mu:.3f}) "
            f"for k={k}, motif_len={motif_len}."
        )
    return mu


def _sample_repeats_geometric(
    rng: np.random.Generator,
    mean_repeats: float,
    *,
    cap: int,
) -> int:
    """
    Sample repeats R ~ Geometric(p) on {1,2,...} with E[R]=1/p = mean_repeats,
    then clamp to `cap`.
    """
    p = 1.0 / float(mean_repeats)
    # numeric guard: p must be in (0,1]
    p = min(max(p, 1e-12), 1.0)
    r = int(rng.geometric(p))
    return min(r, int(cap))


def _generate_class_labels(
    cfg: SyntheticTandemRepeatConfig,
    *,
    seed: int,
) -> list[tuple[str, ...]]:
    """
    Generate `n_classes` unique motif-tuples (labels). Deterministic given seed.

    - k is drawn uniformly from [motifs_per_label_min, motifs_per_label_max]
    - each motif length is drawn uniformly from [motif_len_min, motif_len_max]
    - motifs are sampled uniformly over cfg.alphabet and are distinct within a label
    - label tuples are unique across classes
    """
    rng = np.random.default_rng(seed)

    classes: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    attempts = 0
    max_attempts = 100_000

    while len(classes) < cfg.n_classes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to generate unique class labels; relax constraints."
            )

        k = _rand_int_inclusive(rng, cfg.motifs_per_label_min, cfg.motifs_per_label_max)

        motifs: list[str] = []
        motif_set: set[str] = set()
        for _ in range(k):
            # try a few times to avoid duplicates within the tuple
            for _try in range(50):
                mlen = _rand_int_inclusive(rng, cfg.motif_len_min, cfg.motif_len_max)
                m = _random_string(rng, cfg.alphabet, mlen)
                if m not in motif_set:
                    motif_set.add(m)
                    motifs.append(m)
                    break
            else:
                # could not find distinct motif quickly; restart this class draw
                motifs = []
                break

        if not motifs:
            continue

        label = tuple(motifs)
        if label not in seen:
            seen.add(label)
            classes.append(label)

    return classes


def _label_to_string(label: tuple[str, ...], sep: str) -> str:
    return sep.join(label)


def generate_synthetic_tandem_repeat_df(
    cfg: SyntheticTandemRepeatConfig,
    *,
    seed_labels: int,
    seed_samples: int,
) -> pl.DataFrame:
    """
    Generate a Polars DataFrame with columns:
      - label: string (motif-tuple rendered with cfg.label_sep)
      - sequence: string

    The class definitions (motif tuples) are controlled by seed_labels,
    and the per-sample randomness is controlled by seed_samples.
    """
    if cfg.n_samples <= 0 or cfg.n_classes <= 0:
        raise ValueError("n_samples and n_classes must be positive.")

    if cfg.n_samples % cfg.n_classes != 0:
        raise ValueError(
            "For reproducibility and balanced classes, require n_samples divisible by n_classes."
        )

    class_labels = _generate_class_labels(cfg, seed=seed_labels)
    rng = np.random.default_rng(seed_samples)

    n_per_class = cfg.n_samples // cfg.n_classes

    labels_out: list[str] = []
    seqs_out: list[str] = []

    for class_label in class_labels:
        k = len(class_label)
        label_str = _label_to_string(class_label, cfg.label_sep)

        for _ in range(n_per_class):
            # Prefix / suffix
            pre_len = _rand_int_inclusive(rng, cfg.flank_len_min, cfg.flank_len_max)
            suf_len = _rand_int_inclusive(rng, cfg.flank_len_min, cfg.flank_len_max)

            prefix = _random_string(rng, cfg.alphabet, pre_len)
            suffix = _random_string(rng, cfg.alphabet, suf_len)

            # Separators between motifs
            seps: list[str] = []
            for _j in range(max(k - 1, 0)):
                sep_len = _rand_int_inclusive(rng, cfg.sep_len_min, cfg.sep_len_max)
                seps.append(_random_string(rng, cfg.alphabet, sep_len))

            # Repeat blocks
            blocks: list[str] = []
            for motif in class_label:
                if cfg.coerce_equal_length:
                    mu = _mean_repeats_for_class_and_motif(
                        cfg, k=k, motif_len=len(motif)
                    )
                    r = _sample_repeats_geometric(rng, mu, cap=cfg.repeat_cap)
                else:
                    # Uniform repeats => lengths can vary wildly (intentionally)
                    r = _rand_int_inclusive(rng, 1, cfg.repeat_cap)

                blocks.append(motif * r)

            # Assemble: prefix + block1 + sep1 + block2 + ... + blockK + suffix
            parts: list[str] = [prefix]
            for j in range(k):
                parts.append(blocks[j])
                if j < k - 1:
                    parts.append(seps[j])
            parts.append(suffix)
            seq = "".join(parts)

            # Optional mutations in non-motif regions only:
            # We mutate prefix, separators, suffix (not the motif blocks).
            if cfg.mutation_rate_non_motif > 0:
                prefix_m = _mutate_string_nonmotif(
                    rng, prefix, cfg.alphabet, cfg.mutation_rate_non_motif
                )
                suffix_m = _mutate_string_nonmotif(
                    rng, suffix, cfg.alphabet, cfg.mutation_rate_non_motif
                )
                seps_m = [
                    _mutate_string_nonmotif(
                        rng, s, cfg.alphabet, cfg.mutation_rate_non_motif
                    )
                    for s in seps
                ]
                parts2: list[str] = [prefix_m]
                for j in range(k):
                    parts2.append(blocks[j])  # motif blocks unchanged
                    if j < k - 1:
                        parts2.append(seps_m[j])
                parts2.append(suffix_m)
                seq = "".join(parts2)

            labels_out.append(label_str)
            seqs_out.append(seq)

    return pl.DataFrame({"label": labels_out, "sequence": seqs_out})


def ensure_synthetic_tandem_repeat_dataset(
    out_path: str | Path,
    cfg: SyntheticTandemRepeatConfig = DEFAULT_SYNTHETIC_TR_CONFIG,
    *,
    seed_labels: int = 0,
    seed_samples: int = 1,
) -> pl.DataFrame:
    """
    Read dataset from `out_path` if it exists; otherwise generate and write parquet.

    This makes it easy to plug into DataImporter without regenerating
    new random data every run.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return pl.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_tandem_repeat_df(
        cfg, seed_labels=seed_labels, seed_samples=seed_samples
    )
    df.write_parquet(out_path)
    return df
