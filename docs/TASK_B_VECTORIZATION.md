# Task B — SPMD structure, SIMD vectorization, and core affinity

## SPMD + vectorized loop structure

Task B uses **SPMD** at the row level and **SIMD** on the innermost `x` dimension:

1. `#pragma omp parallel for` — each thread executes the same row program on different `y`.
2. Per-row accumulators `acc0/acc1/acc2[x]` (private to the thread).
3. `#pragma omp simd safelen(64)` on the interior `x` loop — independent pixels, stride-1 access on `src.rgb`.
4. Scalar **peel loops** for left/right borders where clamping would inhibit vectorization.

Gaussian blur precomputes a **2D kernel** (`kernel2d[ky,kx] = kernel1d[ky]·kernel1d[kx]`) so the inner SIMD loop is a tight multiply-add over `x`.

## Verify vectorization (GCC)

```powershell
.\scripts\build_vectorized.ps1
```

This builds with `-O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed` and writes `build/vectorization_report.log`.

GCC report (`build/vectorization_report.log`) on this codebase confirms:

| Location | Result |
|----------|--------|
| `main.cpp:440` (`#pragma omp simd` interior `x`, Gaussian) | **loop vectorized** — 64-byte and 32-byte vectors (AVX-512 / AVX2) |
| `main.cpp:404` (border peel `x`) | **loop vectorized** — 64-byte and 16-byte vectors |
| `main.cpp:478` (`convolve_sobel` `x` loop) | partial vectorization — 8/16-byte vectors |

The outer `ky`/`kx` nests are not fully vectorized (multi-loop nest); the **innermost SPMD `x` loop** is the vectorized hot path.

Compare against the scalar reference `convolve_2d_gaussian_scalar` (not used in the pipeline; kept for correctness checks).

## Core binding (`OMP_PROC_BIND`, `OMP_PLACES`)

OpenMP affinity is set **before** the process starts (each benchmark run is a new process):

```powershell
$env:OMP_NUM_THREADS = 16
.\scripts\benchmark_affinity.ps1
```

Writes `benchmark_affinity.csv` with Task B time per configuration:

| Label | Typical settings |
|-------|------------------|
| `default` | No binding |
| `bind_false` | `OMP_PROC_BIND=false` |
| `bind_true_cores` | `OMP_PROC_BIND=true`, `OMP_PLACES=cores` |
| `bind_close_cores` | `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `bind_spread_cores` | `OMP_PROC_BIND=spread`, `OMP_PLACES=cores` |
| `bind_close_threads` | `OMP_PROC_BIND=close`, `OMP_PLACES=threads` |

If `perf` is available (Linux/WSL), the script also records:

- **L1 miss rate** ≈ `L1-dcache-load-misses / L1-dcache-loads`
- **LLC miss rate** ≈ `cache-misses / cache-references`

### Interpreting results

- **`bind_close` + `places=cores`**: threads stay on fixed physical cores → better L1/L2 reuse of row buffers (`acc0/1/2`) and image rows.
- **`spread`**: spreads threads for bandwidth; can increase cache misses on shared image data.
- **No binding**: OS migration between cores → more cold L1/L2 lines.

Fill in your `benchmark_affinity.csv` and cite whether **lower time** correlates with **lower L1/LLC miss %**.
