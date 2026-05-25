# Color histogram — synchronization and false sharing

The final filtered image uses an **RGB888 histogram**: each pixel maps to one of `2²⁴ = 16 777 216` bins via `(R << 16) | (G << 8) | B`.

## Implementations

| Method | Strategy | Shared writes during pixel loop |
|--------|----------|----------------------------------|
| `critical` | `#pragma omp critical` on `++hist[idx]` | Yes — serialized |
| `atomic` | `#pragma omp atomic update` on `hist[idx]` | Yes — lock-free, cache traffic |
| `local_sparse` | Private `unordered_map` per thread, merge into dense `hist[]` | No |
| `local_dense` | Private full 128 MiB vector per thread (optional, `OMP_HISTOGRAM_DENSE=1`) | No |
| `shared_counters_packed` | Adjacent `uint64_t` per thread + `atomic` (micro-benchmark) | Yes |
| `shared_counters_padded` | `alignas(64)` counter per thread + `atomic` | Yes, isolated lines |

`local_dense` needs ~128 MiB × thread count RAM (e.g. 2 GiB for 16 threads) and may fail on some systems; the default production path is **`local_sparse`**.

## Mutual exclusion vs local accumulation

**Critical** and **atomic** update a **single shared** `hist[]`. Different threads often increment **different bins**, but bins that fall in the same **64-byte cache line** still cause **false sharing**: cores invalidate each other’s lines even without a logical race.

**Local sparse maps** (and dense private histograms) accumulate in **thread-private** structures; the shared dense array is written only during the **merge** (serial or by bin), so the hot loop avoids synchronization.

Typical ordering (fastest → slowest):

1. `local_sparse`
2. `atomic`
3. `critical`

Run:

```bash
$env:OMP_NUM_THREADS=16
.\build\fractal_convolve_parallel.exe --benchmark-histogram
```

Results: `benchmark_histogram.csv`.

## False sharing — did it occur?

### 1. Shared `hist[]` with `atomic` / `critical`

**Yes.** The histogram array is dense. Neighbouring RGB indices often map to neighbouring `hist[idx]` values in the same cache line. Concurrent `atomic`/`critical` updates to those bins produce **cache-line ping-pong**. This is the main false-sharing effect when “saving data to shared arrays” during histogram construction.

Evidence: `atomic` is usually faster than `critical` (less serialization) but still much slower than `local_sparse` because of cache coherence traffic on `hist[]`.

### 2. Private per-thread data (`local_sparse`)

**No** false sharing in the parallel loop: each thread writes only its own hash table. The final merge touches the shared dense array **once per non-zero bin**, not per pixel.

### 3. Packed vs padded thread counters (controlled experiment)

`shared_counters_packed` places all thread counters in consecutive `uint64_t` slots (same cache line for several threads). `shared_counters_padded` places each counter on its own cache line (`alignas(64)`).

If `packed_time / padded_time > 1`, **false sharing is confirmed** on the shared counter array. That isolates the phenomenon independently of the 16 M-bin histogram size.

### 4. Dense private histograms (`local_dense`)

Regions are 128 MiB per thread and **cache-line-aligned** at block boundaries, so the fill phase does **not** false-share across threads. This mode is optional due to memory cost.

## Measured on this machine (16 threads, 8K blurred image)

| Method | Time (s) | vs `critical` | Correct |
|--------|----------|---------------|---------|
| critical | 0.970 | 1.00× | yes |
| atomic | 0.047 | 20.4× | yes |
| local_sparse | 0.048 | 20.1× | yes |
| shared_counters_packed | 0.184 | 5.3× | yes |
| shared_counters_padded | 0.065 | 15.0× | yes |

Packed/padded counter ratio: **2.84×** → false sharing confirmed on adjacent shared counters.

**Conclusion:** `critical` serializes every update and is slowest. `atomic` and `local_sparse` tie (~0.05 s) on this run; both avoid the critical bottleneck. False sharing is visible in the **packed vs padded** experiment (2.84×) and explains part of the gap between `critical` and `atomic` on the shared `hist[]` array.
