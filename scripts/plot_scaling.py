#!/usr/bin/env python3
"""Plot Task A time and speedup vs thread count with Amdahl fit and overhead markers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "benchmark_threads.csv"
OUT_TIME = ROOT / "graphs" / "task_a_time_vs_threads.png"
OUT_SPEEDUP = ROOT / "graphs" / "task_a_speedup_vs_threads.png"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    threads, times, speedups, logical = [], [], [], 16
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threads.append(int(row["threads"]))
            times.append(float(row["seconds"]))
            speedups.append(float(row["speedup"]))
            logical = int(row.get("logical_cores") or logical)
    return (
        np.array(threads, dtype=float),
        np.array(times, dtype=float),
        np.array(speedups, dtype=float),
        logical,
    )


def amdahl_speedup(p: np.ndarray, s: float) -> np.ndarray:
    return 1.0 / (s + (1.0 - s) / p)


def fit_serial_fraction(threads: np.ndarray, speedups: np.ndarray, logical: int) -> float:
    mask = (threads >= 1) & (threads <= logical)
    t = threads[mask]
    s_meas = speedups[mask]
    s_grid = np.linspace(1e-5, 0.95, 8000)
    errors = [np.sum((s_meas - amdahl_speedup(t, s)) ** 2) for s in s_grid]
    return float(s_grid[int(np.argmin(errors))])


def find_overhead_point(
    threads: np.ndarray, times: np.ndarray, logical: int
) -> tuple[int, int, float]:
    """Return (n_opt, n_overhead, t_opt). n_overhead = first increase after best time."""
    n_opt = int(threads[np.argmin(times)])
    t_opt = float(np.min(times))

    n_overhead = int(threads[-1])
    for i in range(1, len(threads)):
        n = int(threads[i])
        if n > n_opt and times[i] > times[i - 1]:
            n_overhead = n
            break

    return n_opt, n_overhead, t_opt


def main() -> None:
    threads, times, speedups, logical = load_data(CSV_PATH)
    max_threads = int(threads[-1])

    s = fit_serial_fraction(threads, speedups, logical)
    s_limit = 1.0 / s

    n_opt, n_overhead, t_opt = find_overhead_point(threads, times, logical)
    t_overhead = float(times[threads == n_overhead][0])

    p_fit = np.linspace(1, max_threads, 400)
    amdahl_curve = amdahl_speedup(p_fit, s)

    OUT_TIME.parent.mkdir(parents=True, exist_ok=True)

    # --- Time vs threads ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(threads, times, "o-", color="#2563eb", linewidth=2, markersize=6, label="Measured")
    ax.axvline(logical, color="#16a34a", linestyle="--", linewidth=1.5, label=f"Logical cores ({logical})")
    ax.axhline(t_opt, color="#9333ea", linestyle=":", linewidth=1.5,
               label=f"Amdahl floor ≈ {t_opt:.3f} s (T₁·s)")
    ax.axvline(n_overhead, color="#dc2626", linestyle="-.", linewidth=2,
               label=f"OS overhead onset (thread {n_overhead})")
    ax.scatter([n_opt], [t_opt], s=120, c="#16a34a", zorder=5, edgecolors="black",
               label=f"Best time @ {n_opt} threads")
    ax.scatter([n_overhead], [t_overhead], s=120, c="#dc2626", zorder=5, edgecolors="black")

    ax.annotate(
        f"Overhead: {n_overhead} threads\n({t_overhead:.3f} s > {times[threads == n_overhead - 1][0]:.3f} s)",
        xy=(n_overhead, t_overhead),
        xytext=(n_overhead + 2, t_overhead + 0.15),
        arrowprops=dict(arrowstyle="->", color="#dc2626"),
        fontsize=9,
    )
    ax.annotate(
        f"Amdahl serial fraction\ns ≈ {s:.3f} → T_min ≈ T₁·s",
        xy=(logical * 0.55, t_opt),
        fontsize=9,
        color="#9333ea",
    )

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Task A: execution time vs threads (Mandelbrot 8K, static chunk=16)")
    ax.set_xticks(np.arange(1, max_threads + 1, 2 if max_threads > 20 else 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_TIME, dpi=150)
    plt.close(fig)

    # --- Speedup vs threads ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(threads, speedups, "o-", color="#2563eb", linewidth=2, markersize=6, label="Measured speedup")
    ax.plot(p_fit, amdahl_curve, color="#9333ea", linewidth=2, linestyle="--",
            label=f"Amdahl fit (s={s:.3f})")
    ax.axhline(s_limit, color="#9333ea", linestyle=":", linewidth=1.5,
               label=f"Amdahl limit 1/s ≈ {s_limit:.2f}×")
    ax.axvline(logical, color="#16a34a", linestyle="--", linewidth=1.5, label=f"Logical cores ({logical})")
    ax.axvline(n_overhead, color="#dc2626", linestyle="-.", linewidth=2,
               label=f"OS overhead onset (thread {n_overhead})")

    s_at_overhead = float(speedups[threads == n_overhead][0])
    ax.scatter([n_overhead], [s_at_overhead], s=120, c="#dc2626", zorder=5, edgecolors="black")

    ax.annotate(
        f"Limit ≈ {s_limit:.2f}×",
        xy=(max_threads * 0.72, s_limit),
        xytext=(max_threads * 0.55, s_limit + 0.4),
        arrowprops=dict(arrowstyle="->", color="#9333ea"),
        fontsize=9,
        color="#9333ea",
    )

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Speedup (T₁ / Tₙ)")
    ax.set_title("Task A: speedup vs threads")
    ax.set_xticks(np.arange(1, max_threads + 1, 2 if max_threads > 20 else 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_SPEEDUP, dpi=150)
    plt.close(fig)

    print(f"Serial fraction s = {s:.4f}")
    print(f"Amdahl speedup limit 1/s = {s_limit:.3f}x")
    print(f"Best threads: {n_opt}  (t = {t_opt:.3f} s)")
    print(f"OS overhead onset: thread {n_overhead}  (t = {t_overhead:.3f} s)")
    print(f"Wrote {OUT_TIME}")
    print(f"Wrote {OUT_SPEEDUP}")


if __name__ == "__main__":
    main()
