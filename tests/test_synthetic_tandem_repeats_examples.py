from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest

import src.data.synthetic_tandem_repeats as tr


def _base_cfg(**overrides: object) -> tr.SyntheticTandemRepeatConfig:
    """
    Small, valid config for fast tests. Override fields via kwargs.
    """
    cfg = tr.SyntheticTandemRepeatConfig(
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
        repeat_cap=20,
        mutation_rate_non_motif=0.0,
        label_sep="|",
    )
    return tr.SyntheticTandemRepeatConfig(**{**cfg.__dict__, **overrides})


def test_rand_int_inclusive_bounds_and_error() -> None:
    rng = np.random.default_rng(0)
    assert tr._rand_int_inclusive(rng, 5, 5) == 5
    x = tr._rand_int_inclusive(rng, 2, 4)
    assert 2 <= x <= 4

    with pytest.raises(ValueError):
        tr._rand_int_inclusive(rng, 3, 2)


def test_random_string_length_nonpositive() -> None:
    rng = np.random.default_rng(0)
    assert tr._random_string(rng, "ACGT", 0) == ""
    assert tr._random_string(rng, "ACGT", -7) == ""


def test_mutate_string_nonmotif_noop_paths() -> None:
    rng = np.random.default_rng(0)
    assert tr._mutate_string_nonmotif(rng, "", "ACGT", 1.0) == ""
    assert tr._mutate_string_nonmotif(rng, "ACGT", "ACGT", 0.0) == "ACGT"
    assert tr._mutate_string_nonmotif(rng, "ACGT", "ACGT", -1.0) == "ACGT"


def test_mutate_string_nonmotif_full_mutation_changes_every_position() -> None:
    rng = np.random.default_rng(0)
    s = "AACCGGTT"
    mutated = tr._mutate_string_nonmotif(rng, s, "ACGT", 1.0)
    assert len(mutated) == len(s)
    # Guaranteed by implementation: substituted symbol is different from original
    assert all(a != b for a, b in zip(s, mutated))
    assert set(mutated) <= set("ACGT")


def test_expected_uniform_int_formula() -> None:
    assert tr._expected_uniform_int(0, 0) == 0.0
    assert tr._expected_uniform_int(0, 2) == 1.0
    assert tr._expected_uniform_int(3, 7) == 5.0


def test_mean_repeats_errors_target_too_small_and_mu_lt_1() -> None:
    # Case 1: target too small relative to expected flanks
    cfg1 = _base_cfg(
        flank_len_min=5,
        flank_len_max=5,
        sep_len_min=0,
        sep_len_max=0,
        target_expected_total_length=5.0,  # expected flanks alone are 10
    )
    with pytest.raises(ValueError, match="target_expected_total_length too small"):
        tr._mean_repeats_for_class_and_motif(cfg1, k=1, motif_len=1)

    # Case 2: motif budget positive, but mu < 1
    cfg2 = _base_cfg(
        flank_len_min=0,
        flank_len_max=0,
        sep_len_min=0,
        sep_len_max=0,
        target_expected_total_length=1.0,  # motif_budget_total=1, but motif_len=2 -> mu=0.5
    )
    with pytest.raises(ValueError, match="mean repeats < 1"):
        tr._mean_repeats_for_class_and_motif(cfg2, k=1, motif_len=2)


def test_sample_repeats_geometric_clamps_p_and_cap() -> None:
    rng = np.random.default_rng(0)

    # mean_repeats < 1 => p > 1, should clamp to 1 => geometric(1) == 1 always
    r1 = tr._sample_repeats_geometric(rng, mean_repeats=0.5, cap=999)
    assert r1 == 1

    # mean_repeats huge => p tiny, clamps to >=1e-12; cap forces deterministic output
    r2 = tr._sample_repeats_geometric(rng, mean_repeats=1e30, cap=1)
    assert r2 == 1

    # Sanity: always >= 1 and <= cap
    r3 = tr._sample_repeats_geometric(rng, mean_repeats=10.0, cap=7)
    assert 1 <= r3 <= 7


def test_label_to_string() -> None:
    assert tr._label_to_string(("AA", "C", "TT"), sep="|") == "AA|C|TT"
    assert tr._label_to_string((), sep="|") == ""


def test_generate_class_labels_deterministic_unique_and_within_constraints() -> None:
    cfg = _base_cfg(
        n_classes=7,
        motifs_per_label_min=1,
        motifs_per_label_max=3,
        motif_len_min=1,
        motif_len_max=3,
    )
    labels1 = tr._generate_class_labels(cfg, seed=123)
    labels2 = tr._generate_class_labels(cfg, seed=123)
    assert labels1 == labels2

    assert len(labels1) == cfg.n_classes
    assert len(set(labels1)) == cfg.n_classes

    for lab in labels1:
        assert cfg.motifs_per_label_min <= len(lab) <= cfg.motifs_per_label_max
        assert len(set(lab)) == len(lab)  # distinct within label
        for m in lab:
            assert cfg.motif_len_min <= len(m) <= cfg.motif_len_max
            assert set(m) <= set(cfg.alphabet)


def test_generate_class_labels_hits_intra_label_duplicate_restart_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Force the inner 'for _try in range(50)' to fail once (duplicate motif 50 times),
    then succeed on the next outer attempt. This covers the 'else: motifs=[]; break'
    restart branch deterministically without heavy looping.
    """
    cfg = _base_cfg(
        n_classes=1,
        motifs_per_label_min=2,
        motifs_per_label_max=2,  # k=2 fixed
        motif_len_min=1,
        motif_len_max=1,  # motifs are single chars
        alphabet="ACGT",
    )

    call_count = {"n": 0}

    def rigged_random_string(
        rng: np.random.Generator, alphabet: str, length: int
    ) -> str:
        # Attempt 1: first motif = "A" (call 0), then second motif tries 50 times = "A" (calls 1..50)
        # Attempt 2: first motif = "A" (call 51), second motif = "C" (call 52) => success
        n = call_count["n"]
        call_count["n"] += 1
        if n <= 51:
            return "A"
        if n == 52:
            return "C"
        return "G"

    monkeypatch.setattr(tr, "_random_string", rigged_random_string, raising=True)

    out = tr._generate_class_labels(cfg, seed=0)
    assert len(out) == 1
    assert out[0] == ("A", "C")


def test_generate_class_labels_raises_runtime_error_when_insufficient_label_space() -> (
    None
):
    """
    Alphabet size 1 and motif length fixed => only one possible label when k=1.
    Asking for 2 classes forces the max_attempts RuntimeError path (fast, because each attempt is cheap).
    """
    cfg = _base_cfg(
        n_classes=2,
        motifs_per_label_min=1,
        motifs_per_label_max=1,
        alphabet="A",
        motif_len_min=1,
        motif_len_max=1,
    )
    with pytest.raises(RuntimeError, match="Failed to generate unique class labels"):
        tr._generate_class_labels(cfg, seed=0)


def test_generate_df_validations() -> None:
    cfg_bad1 = _base_cfg(n_samples=0)
    with pytest.raises(ValueError, match="n_samples and n_classes must be positive"):
        tr.generate_synthetic_tandem_repeat_df(cfg_bad1, seed_labels=0, seed_samples=0)

    cfg_bad2 = _base_cfg(n_samples=10, n_classes=3)  # not divisible
    with pytest.raises(ValueError, match="n_samples divisible by n_classes"):
        tr.generate_synthetic_tandem_repeat_df(cfg_bad2, seed_labels=0, seed_samples=0)


def test_generate_df_shape_balanced_labels_and_basic_content() -> None:
    cfg = _base_cfg(n_samples=12, n_classes=3, mutation_rate_non_motif=0.0)
    df = tr.generate_synthetic_tandem_repeat_df(cfg, seed_labels=1, seed_samples=2)

    assert df.columns == ["label", "sequence"]
    assert df.height == cfg.n_samples

    labels = df["label"].to_list()
    seqs = df["sequence"].to_list()
    assert len(labels) == len(seqs) == cfg.n_samples

    # Balanced by construction: exactly n_samples/n_classes per label
    n_per = cfg.n_samples // cfg.n_classes
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    assert len(counts) == cfg.n_classes
    assert set(counts.values()) == {n_per}

    # Alphabet-only content and each motif appears in its sequence
    alphabet_set = set(cfg.alphabet)
    for lab, seq in zip(labels, seqs):
        assert set(seq) <= alphabet_set
        motifs = [m for m in lab.split(cfg.label_sep) if m]
        for m in motifs:
            assert m in seq  # each motif block is present at least once


def test_generate_df_mutation_affects_only_non_motif_regions_when_k1_and_mu1() -> None:
    """
    Choose parameters so repeats are deterministically 1:
      - k = 1 fixed
      - motif_len fixed (4)
      - flanks fixed (5 each)
      - target_expected_total_length chosen so mu = 1 exactly
    Then, with mutation_rate=1.0, prefix+suffix should change at every position,
    while the motif block remains identical.
    """
    cfg_base = tr.SyntheticTandemRepeatConfig(
        n_samples=2,
        n_classes=1,
        motifs_per_label_min=1,
        motifs_per_label_max=1,
        alphabet="ACGT",
        motif_len_min=4,
        motif_len_max=4,
        flank_len_min=5,
        flank_len_max=5,
        sep_len_min=0,
        sep_len_max=0,
        target_expected_total_length=14.0,  # 10 flanks + 4 motif budget => mu = 1
        repeat_cap=999,
        mutation_rate_non_motif=0.0,
        label_sep="|",
    )

    df0 = tr.generate_synthetic_tandem_repeat_df(
        cfg_base, seed_labels=7, seed_samples=9
    )

    cfg_mut = tr.SyntheticTandemRepeatConfig(
        **{**cfg_base.__dict__, "mutation_rate_non_motif": 1.0}
    )
    df1 = tr.generate_synthetic_tandem_repeat_df(cfg_mut, seed_labels=7, seed_samples=9)

    motif = df0["label"][0]  # only one class, label is the motif itself (k=1)
    assert isinstance(motif, str)
    assert len(motif) == 4

    s0 = df0["sequence"][0]
    s1 = df1["sequence"][0]
    assert len(s0) == len(s1) == 5 + 4 + 5

    # motif block (middle) unchanged
    assert s0[5:9] == motif
    assert s1[5:9] == motif

    # prefix/suffix changed at every position (guaranteed by mutation logic)
    assert all(a != b for a, b in zip(s0[:5], s1[:5]))
    assert all(a != b for a, b in zip(s0[9:], s1[9:]))


def _df_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.schema == b.schema and a.to_dicts() == b.to_dicts()


def test_ensure_dataset_reads_if_exists_and_generates_if_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "synthetic.parquet"

    # --- read-existing path
    expected = pl.DataFrame({"label": ["x"], "sequence": ["ACGT"]})
    out.write_text("marker")  # make it "exist"

    def fake_read_parquet(path: str | Path) -> pl.DataFrame:
        assert Path(path) == out
        return expected

    monkeypatch.setattr(tr.pl, "read_parquet", fake_read_parquet, raising=True)

    got = tr.ensure_synthetic_tandem_repeat_dataset(out)
    assert _df_equal(got, expected)

    # --- generate-and-write path
    out2 = tmp_path / "synthetic2.parquet"

    generated = pl.DataFrame({"label": ["y"], "sequence": ["TTTT"]})

    def fake_generate(
        cfg: tr.SyntheticTandemRepeatConfig, *, seed_labels: int, seed_samples: int
    ) -> pl.DataFrame:
        assert seed_labels == 11
        assert seed_samples == 22
        return generated

    def fake_write_parquet(self: pl.DataFrame, path: str | Path) -> None:
        Path(path).write_text("written")

    monkeypatch.setattr(
        tr, "generate_synthetic_tandem_repeat_df", fake_generate, raising=True
    )
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write_parquet, raising=True)

    got2 = tr.ensure_synthetic_tandem_repeat_dataset(
        out2, seed_labels=11, seed_samples=22
    )
    assert _df_equal(got2, generated)
    assert out2.exists()
