from __future__ import annotations

from typing import Optional

from hypothesis import given, settings, strategies as st
import polars as pl

from src.clustering.suite import (
    DataImporter,
    DistanceSpec,
    DbscanConfig,
    run_dbscan_suite,
)


def _len_distance(a: str, b: str) -> float:
    return float(abs(len(a) - len(b)))


def _first_char_distance(a: str, b: str) -> float:
    if not a or not b:
        return float(a != b)
    return 0.0 if a[0] == b[0] else 1.0


LABELS = st.sampled_from(["EI", "IE", "N"])
DNA = st.text(alphabet="ACGT", min_size=1, max_size=20)


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(st.tuples(LABELS, DNA), min_size=2, max_size=30),
    max_rows=st.one_of(st.none(), st.integers(min_value=1, max_value=30)),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_run_dbscan_suite_row_count_and_schema_invariants(
    rows: list[tuple[str, str]],
    max_rows: Optional[int],
    seed: int,
) -> None:
    df = pl.DataFrame(
        {
            "label": [lbl for (lbl, _seq) in rows],
            "sequence": [seq for (_lbl, seq) in rows],
        }
    )

    def load_a() -> pl.DataFrame:
        return df

    def load_b() -> pl.DataFrame:
        return df

    importers = [
        DataImporter(name="toy_a", load=load_a),
        DataImporter(name="toy_b", load=load_b),
    ]

    distances = [
        DistanceSpec(name="len", fn=_len_distance),
        DistanceSpec(name="first_char", fn=_first_char_distance),
    ]

    cfg = DbscanConfig(tune=False, eps=0.5, min_samples=2)

    result = run_dbscan_suite(
        importers=importers,
        distances=distances,
        dbscan=cfg,
        max_rows=max_rows,
        seed=seed,
        out_dir=None,  # avoid writes
        make_plots=False,  # avoid viz calls
    )

    assert result.height == len(importers) * len(distances)

    expected_cols = {
        "dataset",
        "distance",
        "n",
        "t_pairwise_seconds",
        "tuned",
        "eps",
        "min_samples",
        "silhouette",
        "ari",
        "nmi",
        "n_clusters_ex_noise",
        "n_noise",
        "heatmap_path",
        "mds_true_path",
        "mds_pred_path",
    }
    assert set(result.columns) == expected_cols

    expected_n = len(rows) if max_rows is None else min(len(rows), int(max_rows))

    assert set(result["dataset"].to_list()) == {"toy_a", "toy_b"}
    assert set(result["distance"].to_list()) == {"len", "first_char"}

    assert all(int(x) == expected_n for x in result["n"].to_list())
    assert all(bool(x) is False for x in result["tuned"].to_list())
    assert all(float(x) == cfg.eps for x in result["eps"].to_list())
    assert all(int(x) == cfg.min_samples for x in result["min_samples"].to_list())

    assert all(x is None for x in result["heatmap_path"].to_list())
    assert all(x is None for x in result["mds_true_path"].to_list())
    assert all(x is None for x in result["mds_pred_path"].to_list())
