from __future__ import annotations

from typing import Tuple

import numpy as np
import polars as pl
from hypothesis import given, settings, strategies as st

import src.data.synthetic_tandem_repeats as tr


def _cfg_small(
    mutation_rate: float = 0.0, *, coerce_equal_length: bool = True
) -> tr.SyntheticTandemRepeatConfig:
    # Keep this stable so Hypothesis runs are fast and non-flaky.
    return tr.SyntheticTandemRepeatConfig(
        n_samples=12,
        n_classes=3,
        motifs_per_label_min=1,
        motifs_per_label_max=3,
        alphabet="ACGT",
        motif_len_min=1,
        motif_len_max=3,
        flank_len_min=0,
        flank_len_max=2,
        sep_len_min=0,
        sep_len_max=2,
        target_expected_total_length=30.0,
        coerce_equal_length=bool(coerce_equal_length),
        repeat_cap=20,
        mutation_rate_non_motif=float(mutation_rate),
        label_sep="|",
    )


@settings(max_examples=60, deadline=None)
@given(
    lo=st.integers(min_value=-10, max_value=10),
    hi=st.integers(min_value=-10, max_value=10),
    seed=st.integers(0, 2**31 - 1),
)
def test_rand_int_inclusive_returns_inclusive_bounds(
    lo: int, hi: int, seed: int
) -> None:
    if hi < lo:
        lo, hi = hi, lo
    rng = np.random.default_rng(seed)
    x = tr._rand_int_inclusive(rng, lo, hi)
    assert lo <= x <= hi


@settings(max_examples=30, deadline=None)
@given(
    seed_labels=st.integers(min_value=0, max_value=2**31 - 1),
    seed_samples=st.integers(min_value=0, max_value=2**31 - 1),
    mutation=st.sampled_from([0.0, 1.0]),
    coerce=st.booleans(),
)
def test_generate_df_invariants_balanced_deterministic_alphabet_and_motifs_present(
    seed_labels: int, seed_samples: int, mutation: float, coerce: bool
) -> None:
    cfg = _cfg_small(mutation_rate=mutation, coerce_equal_length=coerce)

    df1 = tr.generate_synthetic_tandem_repeat_df(
        cfg, seed_labels=seed_labels, seed_samples=seed_samples
    )
    df2 = tr.generate_synthetic_tandem_repeat_df(
        cfg, seed_labels=seed_labels, seed_samples=seed_samples
    )

    # Deterministic for fixed seeds
    assert df1["label"].to_list() == df2["label"].to_list()
    assert df1["sequence"].to_list() == df2["sequence"].to_list()

    assert df1.columns == ["label", "sequence"]
    assert df1.height == cfg.n_samples

    labels = df1["label"].to_list()
    seqs = df1["sequence"].to_list()

    # Balanced labels: exactly n_per_class occurrences
    n_per = cfg.n_samples // cfg.n_classes
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    assert len(counts) == cfg.n_classes
    assert set(counts.values()) == {n_per}

    # Alphabet-only sequences and each motif from the label appears in the sequence
    alphabet_set = set(cfg.alphabet)
    for lab, seq in zip(labels, seqs):
        assert set(seq) <= alphabet_set
        motifs = [m for m in lab.split(cfg.label_sep) if m]
        for m in motifs:
            assert m in seq


@settings(max_examples=40, deadline=None)
@given(
    s=st.text(alphabet="ACGT", min_size=0, max_size=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    rate=st.floats(min_value=0.0, max_value=1.0),
)
def test_mutate_string_nonmotif_preserves_length_and_alphabet(
    s: str, seed: int, rate: float
) -> None:
    rng = np.random.default_rng(seed)
    out = tr._mutate_string_nonmotif(rng, s, "ACGT", float(rate))
    assert len(out) == len(s)
    assert set(out) <= set("ACGT")
