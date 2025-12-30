from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering.suite import DbscanConfig, run_dbscan_suite
from src.experiments.dbscan_defaults import DATA_IMPORTERS, DISTANCES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/dbscan_suite"))
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--trials", type=int, default=40)

    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plot-max-n", type=int, default=600)

    args = ap.parse_args()

    cfg = DbscanConfig(
        tune=not args.no_tune,
        eps=args.eps,
        min_samples=args.min_samples,
        n_trials=args.trials,
    )

    results = run_dbscan_suite(
        importers=DATA_IMPORTERS,
        distances=DISTANCES,
        dbscan=cfg,
        max_rows=args.max_rows,
        seed=args.seed,
        out_dir=args.out_dir,
        make_plots=not args.no_plots,
        plot_max_n=args.plot_max_n,
    )

    print(results.sort(["dataset", "distance"]))

def _report_rankings(results) -> None:
    """
    Print per-dataset rankings for ARI and NMI (best -> worst).
    Expects `results` to be a Polars DataFrame with columns:
      - dataset, distance
      - ari, nmi
    """
    import polars as pl

    required = {"dataset", "distance", "ari", "nmi"}
    missing = required - set(results.columns)
    if missing:
        print(f"[rankings] Skipping: results missing columns: {sorted(missing)}")
        return

    # Ensure floats for sorting; tolerate nulls
    df = results.with_columns(
        pl.col("ari").cast(pl.Float64),
        pl.col("nmi").cast(pl.Float64),
    )

    datasets = df.get_column("dataset").unique().to_list()

    for ds in datasets:
        sub = df.filter(pl.col("dataset") == ds)

        print("\n" + "=" * 90)
        print(f"Dataset: {ds}")

        # ARI ranking
        print("\nARI (best -> worst):")
        ari_rank = (
            sub.sort(["ari", "nmi"], descending=[True, True])
            .select(["distance", "ari", "nmi"])
        )
        for i, row in enumerate(ari_rank.iter_rows(named=True), start=1):
            print(f"{i:>2}. {row['distance']:<35}  ari={row['ari']:.4f}  nmi={row['nmi']:.4f}")

        # NMI ranking
        print("\nNMI (best -> worst):")
        nmi_rank = (
            sub.sort(["nmi", "ari"], descending=[True, True])
            .select(["distance", "ari", "nmi"])
        )
        for i, row in enumerate(nmi_rank.iter_rows(named=True), start=1):
            print(f"{i:>2}. {row['distance']:<35}  nmi={row['nmi']:.4f}  ari={row['ari']:.4f}")

        # Optional: show a combined “best overall” by average rank (ARI rank + NMI rank)
        ari_with_rank = sub.with_columns(
            pl.col("ari").rank(method="dense", descending=True).alias("rank_ari")
        )
        both = ari_with_rank.with_columns(
            pl.col("nmi").rank(method="dense", descending=True).alias("rank_nmi")
        ).with_columns(
            ((pl.col("rank_ari") + pl.col("rank_nmi")) / 2.0).alias("avg_rank")
        )

        top = both.sort(["avg_rank", "rank_ari", "rank_nmi"]).select(
            ["distance", "ari", "nmi", "rank_ari", "rank_nmi", "avg_rank"]
        ).head(3)

        print("\nTop-3 by average rank (ARI+NMI):")
        for row in top.iter_rows(named=True):
            print(
                f"- {row['distance']}: ari={row['ari']:.4f}, nmi={row['nmi']:.4f} "
                f"(rank_ari={int(row['rank_ari'])}, rank_nmi={int(row['rank_nmi'])}, avg={row['avg_rank']:.2f})"
            )
    
    _report_rankings(results)




if __name__ == "__main__":
    main()
