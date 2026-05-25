# Fractal + convolution (OpenMP parallel baseline)

OpenMP-parallel C++ pipeline:

1. **Task A** — Render the Mandelbrot set at **8K UHD** (7680×4320).
2. **Task B** — Apply a **heavy 2D convolution** (Gaussian or Sobel), parallelized by row.

## Build

With CMake (recommended):

```bash
cmake -S . -B build
cmake --build build --config Release
```

With g++ (MSYS2 / MinGW):

```bash
g++ -std=c++17 -O2 -fopenmp -Wall -Wextra -o build/fractal_convolve_parallel.exe src/main.cpp
```

## Run full pipeline

```bash
./build/fractal_convolve_parallel
./build/fractal_convolve_parallel sobel
```

## Task A scheduler benchmark

Compares `schedule(static)`, `schedule(dynamic)`, and `schedule(guided)` over chunk sizes  
`1, 2, 4, …, 4320` (median of 3 runs after 1 warmup):

```bash
./build/fractal_convolve_parallel --benchmark-a
```

Writes `benchmark_task_a.csv` and prints the best configuration for your CPU.

Control thread count:

```bash
export OMP_NUM_THREADS=16    # Linux/macOS
set OMP_NUM_THREADS=16       # Windows cmd
$env:OMP_NUM_THREADS=16      # PowerShell
```

### Empirical result (16-thread run, 7680×4320)

| Configuration | Median Task A time |
|---------------|-------------------|
| **static, chunk=16** (best) | **0.548 s** |
| dynamic, chunk=2–4 | ~0.55 s |
| static default (implicit chunk) | 1.120 s |
| guided, chunk=1–64 | ~0.92–0.93 s |

**Conclusion for this CPU:** `schedule(static, 16)` wins. Mandelbrot rows have uneven cost (more iterations inside the set); a modest static chunk balances load without the overhead of `dynamic` or the coarse decay of `guided`. Large chunks (512+) hurt badly because they serialize the heavy central rows onto fewer threads.

Re-run `--benchmark-a` after changing `OMP_NUM_THREADS` or hardware.

## Thread scaling graphs (Amdahl + OS overhead)

Sweep threads from 1 to 2×logical cores (uses optimal `static,16`):

```bash
./build/fractal_convolve_parallel --benchmark-threads
python scripts/plot_scaling.py
```

Outputs in `graphs/`:

- `task_a_time_vs_threads.png`
- `task_a_speedup_vs_threads.png`

### Results on this machine (16 logical cores)

| Metric | Value |
|--------|-------|
| Serial time (1 thread) | 6.25 s |
| Best time (16 threads) | 0.536 s |
| Peak speedup | **11.65×** @ 16 threads |
| Amdahl fit (threads 1–16) | **s ≈ 0.020** → theoretical limit **≈ 49×** |
| OS overhead onset | **Thread 17** (time 0.561 s > 0.536 s at 16 threads) |

Past 16 threads the runtime scheduler oversubscribes logical CPUs; the first measurable slowdown is at **17 threads**. The Amdahl asymptote (`1/s`) is the horizontal cap on the speedup plot; measured speedup stops growing near 16 threads because the serial fraction and synchronization costs dominate before the theoretical 49× limit is reachable.

## Color histogram (Task C)

After filtering, the pipeline builds an **RGB888 histogram** (count per `(R,G,B)` color) using private per-thread bins + merge.

Compare synchronization strategies (mutual exclusion vs local/reduction):

```bash
./build/fractal_convolve_parallel --benchmark-histogram
```

Writes `benchmark_histogram.csv`. Analysis of **false sharing** on shared vs private arrays: [docs/HISTOGRAM.md](docs/HISTOGRAM.md).

## Outputs

- `mandelbrot_8k.ppm` — fractal
- `mandelbrot_8k_blur.ppm` or `mandelbrot_8k_sobel.ppm` — filtered image
- `benchmark_task_a.csv` — scheduler sweep results
- `benchmark_histogram.csv` — histogram method timings
