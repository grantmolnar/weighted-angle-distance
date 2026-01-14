#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIST_RE = {
    "weighted_angle": re.compile(r"^weighted_angle_rho=(?P<rho>[-+0-9.eE]+)$"),
    "kgram_angle": re.compile(r"^kgram_angle_k=(?P<k>\d+)$"),
    "js_kgram": re.compile(r"^jensen_shannon_kgram_k=(?P<k>\d+)$"),
}

EDIT_NAMES = {"levenshtein", "damerau_levenshtein", "lcs"}


def parse_distance(name: str) -> tuple[str, float | int | None]:
    m = DIST_RE["weighted_angle"].match(name)
    if m:
        return ("weighted_angle", float(m.group("rho")))
    m = DIST_RE["kgram_angle"].match(name)
    if m:
        return ("kgram_angle", int(m.group("k")))
    m = DIST_RE["js_kgram"].match(name)
    if m:
        return ("js_kgram", int(m.group("k")))
    if name in EDIT_NAMES:
        return ("edit", None)
    return ("other", None)


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    # Spearman = Pearson correlation of ranks
    ra = a.rank(ascending=False, method="average")
    rb = b.rank(ascending=False, method="average")
    return float(ra.corr(rb))


def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_best_of_family(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    # best score within each (dataset, family)
    best = (
        df.groupby(["dataset", "family"], as_index=False)[metric]
        .max()
        .sort_values(["dataset", metric], ascending=[True, False])
    )

    for ds, sub in best.groupby("dataset"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(sub["family"], sub[metric])
        ax.set_title(f"{ds}: best {metric} by family")
        ax.set_ylabel(metric)
        ax.set_xlabel("family")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}__best_{metric}_by_family.png", dpi=200)
        plt.close(fig)


def plot_param_sweeps(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    # weighted_angle sweeps (rho)
    wa = df[df["family"] == "weighted_angle"].copy()
    if not wa.empty:
        for ds, sub in wa.groupby("dataset"):
            sub = sub.sort_values("param")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sub["param"].astype(float), sub[metric].astype(float), marker="o")
            ax.set_title(f"{ds}: {metric} vs rho (weighted angle)")
            ax.set_xlabel("rho")
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(out_dir / f"{ds}__{metric}__vs_rho.png", dpi=200)
            plt.close(fig)

    # kgram_angle sweeps (k)
    ka = df[df["family"] == "kgram_angle"].copy()
    if not ka.empty:
        for ds, sub in ka.groupby("dataset"):
            sub = sub.sort_values("param")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sub["param"].astype(int), sub[metric].astype(float), marker="o")
            ax.set_title(f"{ds}: {metric} vs k (k-gram angle)")
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(out_dir / f"{ds}__{metric}__vs_k_kgram_angle.png", dpi=200)
            plt.close(fig)

    # js_kgram sweeps (k)
    js = df[df["family"] == "js_kgram"].copy()
    if not js.empty:
        for ds, sub in js.groupby("dataset"):
            sub = sub.sort_values("param")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sub["param"].astype(int), sub[metric].astype(float), marker="o")
            ax.set_title(f"{ds}: {metric} vs k (JS k-gram)")
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(out_dir / f"{ds}__{metric}__vs_k_js_kgram.png", dpi=200)
            plt.close(fig)


def rank_consistency_table(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [m for m in ["ari", "nmi", "silhouette"] if m in df.columns]
    rows = []
    for ds, sub in df.groupby("dataset"):
        sub = sub.set_index("distance")
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                m1, m2 = metrics[i], metrics[j]
                corr = spearman_rank_corr(sub[m1], sub[m2])
                rows.append({"dataset": ds, "metric_a": m1, "metric_b": m2, "spearman_rank_corr": corr})
    out = pd.DataFrame(rows).sort_values(["dataset", "metric_a", "metric_b"])
    out.to_csv(out_dir / "rank_consistency_across_metrics.csv", index=False)


def rho_stability_tables(df: pd.DataFrame, out_dir: Path) -> None:
    wa = df[df["family"] == "weighted_angle"].copy()
    if wa.empty:
        return

    # per-dataset: best rho and variability across rho
    rows = []
    for ds, sub in wa.groupby("dataset"):
        sub = sub.copy()
        sub["rho"] = sub["param"].astype(float)
        for metric in ["ari", "nmi", "silhouette"]:
            if metric not in sub.columns:
                continue
            best_row = sub.loc[sub[metric].astype(float).idxmax()]
            vals = sub[metric].astype(float).to_numpy()
            rows.append({
                "dataset": ds,
                "metric": metric,
                "best_rho": float(best_row["rho"]),
                "best_value": float(best_row[metric]),
                "std_over_rho": float(np.std(vals)),
                "range_over_rho": float(np.max(vals) - np.min(vals)),
            })

    out = pd.DataFrame(rows).sort_values(["dataset", "metric"])
    out.to_csv(out_dir / "weighted_angle_rho_stability.csv", index=False)

    # how often each rho wins (by ARI, and by NMI)
    wins = []
    for metric in ["ari", "nmi"]:
        if metric not in wa.columns:
            continue
        idx = wa.groupby("dataset")[metric].idxmax()
        winners = wa.loc[idx, ["dataset", "param"]].copy()
        winners["rho"] = winners["param"].astype(float)
        counts = winners["rho"].value_counts().sort_index()
        for rho, c in counts.items():
            wins.append({"metric": metric, "rho": float(rho), "n_datasets_won": int(c)})
    if wins:
        pd.DataFrame(wins).to_csv(out_dir / "weighted_angle_rho_win_counts.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("outputs/dbscan_suite/results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/plots_from_results"))
    args = ap.parse_args()

    out_dir = ensure_outdir(args.out_dir)

    df = pd.read_csv(args.results)

    required = {"dataset", "distance"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"results.csv missing required columns: {sorted(missing)}")

    fam_param = df["distance"].astype(str).apply(parse_distance)
    df["family"] = fam_param.apply(lambda t: t[0])
    df["param"] = fam_param.apply(lambda t: t[1])

    # Make a copy with numeric metrics
    for m in ["ari", "nmi", "silhouette", "n_noise", "n", "n_clusters_ex_noise"]:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Core outputs
    for metric in ["ari", "nmi", "silhouette"]:
        if metric in df.columns:
            plot_best_of_family(df, metric, out_dir)
            plot_param_sweeps(df, metric, out_dir)

    rank_consistency_table(df, out_dir)
    rho_stability_tables(df, out_dir)

    # Optional sanity plot: ARI vs noise fraction
    if {"ari", "n_noise", "n"} <= set(df.columns):
        df2 = df.copy()
        df2["noise_frac"] = df2["n_noise"] / df2["n"]
        for ds, sub in df2.groupby("dataset"):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(sub["noise_frac"], sub["ari"])
            ax.set_title(f"{ds}: ARI vs noise fraction")
            ax.set_xlabel("noise fraction (n_noise / n)")
            ax.set_ylabel("ARI")
            fig.tight_layout()
            fig.savefig(out_dir / f"{ds}__ari_vs_noise_frac.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
