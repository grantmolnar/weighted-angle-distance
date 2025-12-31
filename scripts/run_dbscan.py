from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering.suite import DbscanConfig, run_dbscan_suite, report_rankings
from src.experiments.dbscan_defaults import DATA_IMPORTERS, DISTANCES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/dbscan_suite"))
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--trials", type=int, default=100)

    # ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plot-max-n", type=int, default=600)
    ap.add_argument("--max-rows", type=int, default=0, help="Max rows per dataset. Use 0 to use the full dataset.")

    args = ap.parse_args()

    cfg = DbscanConfig(
        tune=not args.no_tune,
        eps=args.eps,
        min_samples=args.min_samples,
        n_trials=args.trials,
    )

    max_rows = None if args.max_rows <= 0 else args.max_rows

    results = run_dbscan_suite(
        importers=DATA_IMPORTERS,
        distances=DISTANCES,
        dbscan=cfg,
        max_rows=max_rows,
        seed=args.seed,
        out_dir=args.out_dir,
        # make_plots=not args.no_plots,
        plot_max_n=args.plot_max_n,
    )

    print(results.sort(["dataset", "distance"]))

    report_rankings(results)


if __name__ == "__main__":
    main()
