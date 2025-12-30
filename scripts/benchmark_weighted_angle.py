from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np


# Sample call:
#  python -m scripts.benchmark_weighted_angle   --rho 0.6180339887498948   --lengths 64,96,128,192,256,384,512,768,1024,1536,2048   --reps 9   --pairs-per-length 4   --out-csv outputs/benchmark_weighted_angle.csv   --plot

# ----------------------------
# 1) Import your two impls here
# ----------------------------
#
# Edit these two lines if your function names differ.
#
# The functions should have signature like: fn(s: str, t: str, rho: float) -> float
#
from src.string_distances.weighted_angle_distance import (  # type: ignore
    naive_weighted_angle_distance as naive_wa,
)
from src.string_distances.weighted_angle_distance import (  # type: ignore
    weighted_angle_distance as trie_wa,
)


# ----------------------------
# Benchmarking helpers
# ----------------------------

@dataclass(frozen=True)
class BenchRow:
    n_s: int
    n_t: int
    rho: float
    reps: int
    naive_s: float
    trie_s: float
    speedup: float
    naive_val: float
    trie_val: float
    abs_err: float


def _rand_dna(rng: np.random.Generator, n: int, alphabet: str) -> str:
    if n <= 0:
        return ""
    idx = rng.integers(0, len(alphabet), size=n)
    return "".join(alphabet[i] for i in idx.tolist())


def _time_median_seconds(fn: Callable[[], float], reps: int) -> Tuple[float, float]:
    """
    Run fn() reps times, returning (median_seconds, last_value).
    """
    times: List[float] = []
    last_val = 0.0
    # keep timing less noisy
    gcold = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            last_val = float(fn())
            t1 = time.perf_counter()
            times.append(t1 - t0)
    finally:
        if gcold:
            gc.enable()
    return statistics.median(times), last_val


def _loglog_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    """
    Fit log(y) = a + b log(x); return b.
    Ignores any nonpositive points (shouldn't happen here).
    """
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    b, _a = np.polyfit(np.log(x), np.log(y), deg=1)
    return float(b)


def _write_csv(path: Path, rows: List[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n_s",
                "n_t",
                "rho",
                "reps",
                "naive_seconds_median",
                "trie_seconds_median",
                "speedup_naive_over_trie",
                "naive_value",
                "trie_value",
                "abs_err",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.n_s,
                    r.n_t,
                    r.rho,
                    r.reps,
                    f"{r.naive_s:.6e}",
                    f"{r.trie_s:.6e}",
                    f"{r.speedup:.6f}",
                    f"{r.naive_val:.12g}",
                    f"{r.trie_val:.12g}",
                    f"{r.abs_err:.3e}",
                ]
            )


def _maybe_plot(path: Path, rows: List[BenchRow]) -> None:
    try:
        import matplotlib.pyplot as plt  # installed in your env
    except Exception:
        print("matplotlib not available; skipping plot.")
        return

    # group by (n_s,n_t) but here we’ll just plot vs n where n_s==n_t in typical runs
    ns = [r.n_s for r in rows]
    naive = [r.naive_s for r in rows]
    trie = [r.trie_s for r in rows]
    speedup = [r.speedup for r in rows]

    # time plot
    plt.figure()
    plt.plot(ns, naive, marker="o", label="naive median seconds")
    plt.plot(ns, trie, marker="o", label="suffix-trie median seconds")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("sequence length n")
    plt.ylabel("time (seconds, median over reps)")
    plt.legend()
    p1 = path.with_suffix(".time.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()

    # speedup plot
    plt.figure()
    plt.plot(ns, speedup, marker="o", label="speedup = naive / trie")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sequence length n")
    plt.ylabel("speedup")
    plt.legend()
    p2 = path.with_suffix(".speedup.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()

    print(f"Wrote plots:\n  {p1}\n  {p2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--alphabet", type=str, default="ACGT")
    ap.add_argument(
        "--lengths",
        type=str,
        default="64,96,128,192,256,384,512,768,1024",
        help="Comma-separated lengths to benchmark (S and T).",
    )
    ap.add_argument("--reps", type=int, default=7, help="Repetitions per length per impl.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pairs-per-length",
        type=int,
        default=3,
        help="How many different (S,T) pairs to average per length.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="outputs/benchmark_weighted_angle.csv",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="If set, write log-log plots (time + speedup).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Correctness tolerance for |naive - trie| (warn if exceeded).",
    )
    args = ap.parse_args()

    rho = float(args.rho)
    alphabet = str(args.alphabet)
    lengths = [int(x.strip()) for x in str(args.lengths).split(",") if x.strip()]
    out_csv = Path(args.out_csv)

    rng = np.random.default_rng(int(args.seed))

    rows: List[BenchRow] = []

    print("Benchmarking weighted-angle distance")
    print(f"  rho={rho}")
    print(f"  alphabet={alphabet}")
    print(f"  lengths={lengths}")
    print(f"  reps={args.reps}  pairs_per_length={args.pairs_per_length}")
    print()

    for n in lengths:
        # average over a few pairs for stability
        naive_times: List[float] = []
        trie_times: List[float] = []
        abs_errs: List[float] = []
        naive_vals: List[float] = []
        trie_vals: List[float] = []

        for _ in range(int(args.pairs_per_length)):
            s = _rand_dna(rng, n, alphabet)
            t = _rand_dna(rng, n, alphabet)

            # warmup (JIT doesn’t apply, but caches / allocations do)
            _ = trie_wa(s, t, rho=rho)
            _ = naive_wa(s, t, rho=rho)

            naive_sec, naive_val = _time_median_seconds(
                lambda: naive_wa(s, t, rho=rho), reps=int(args.reps)
            )
            trie_sec, trie_val = _time_median_seconds(
                lambda: trie_wa(s, t, rho=rho), reps=int(args.reps)
            )

            naive_times.append(naive_sec)
            trie_times.append(trie_sec)
            naive_vals.append(naive_val)
            trie_vals.append(trie_val)
            abs_errs.append(abs(float(naive_val) - float(trie_val)))

        naive_med = float(statistics.median(naive_times))
        trie_med = float(statistics.median(trie_times))
        abs_err = float(statistics.median(abs_errs))
        speedup = float(naive_med / trie_med) if trie_med > 0 else float("inf")

        r = BenchRow(
            n_s=n,
            n_t=n,
            rho=rho,
            reps=int(args.reps),
            naive_s=naive_med,
            trie_s=trie_med,
            speedup=speedup,
            naive_val=float(statistics.median(naive_vals)),
            trie_val=float(statistics.median(trie_vals)),
            abs_err=abs_err,
        )
        rows.append(r)

        flag = ""
        if abs_err > float(args.tol):
            flag = "  (!! value mismatch over tol)"
        print(
            f"n={n:5d}  naive={naive_med:9.3e}s  trie={trie_med:9.3e}s"
            f"  speedup={speedup:7.2f}x  |Δ|={abs_err:.2e}{flag}"
        )

    _write_csv(out_csv, rows)
    print(f"\nWrote CSV: {out_csv}")

    # Estimate scaling: time ~ n^alpha  (for S and T same length)
    ns = [float(r.n_s) for r in rows]
    naive_ts = [float(r.naive_s) for r in rows]
    trie_ts = [float(r.trie_s) for r in rows]
    speedups = [float(r.speedup) for r in rows]

    alpha_naive = _loglog_slope(ns, naive_ts)
    alpha_trie = _loglog_slope(ns, trie_ts)
    alpha_speedup = _loglog_slope(ns, speedups)

    print("\nEstimated log–log slopes (time ~ n^alpha):")
    print(f"  naive alpha ≈ {alpha_naive:.3f}")
    print(f"  trie  alpha ≈ {alpha_trie:.3f}")
    print(f"  speedup alpha ≈ {alpha_speedup:.3f}  (expect ~1 if naive~n^2 and trie~n^1)")

    if args.plot:
        _maybe_plot(out_csv, rows)


if __name__ == "__main__":
    main()
