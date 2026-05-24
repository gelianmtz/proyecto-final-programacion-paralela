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

## Outputs

- `mandelbrot_8k.ppm` — fractal
- `mandelbrot_8k_blur.ppm` or `mandelbrot_8k_sobel.ppm` — filtered image
- `benchmark_task_a.csv` — scheduler sweep results
