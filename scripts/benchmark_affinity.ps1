# Sweep OMP_PROC_BIND / OMP_PLACES and measure Task B (+ optional perf cache counters)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\fractal_convolve_parallel.exe"
$Csv = Join-Path $Root "benchmark_affinity.csv"
$Log = Join-Path $Root "benchmark_affinity.log"

if (-not (Test-Path $Exe)) {
    Write-Error "Build first: scripts\build_vectorized.ps1"
}

$env:OMP_NUM_THREADS = "16"
$env:OMP_DISPLAY_AFFINITY = "false"

$configs = @(
    @{ bind = ""; places = ""; label = "default" },
    @{ bind = "false"; places = ""; label = "bind_false" },
    @{ bind = "true"; places = "cores"; label = "bind_true_cores" },
    @{ bind = "close"; places = "cores"; label = "bind_close_cores" },
    @{ bind = "spread"; places = "cores"; label = "bind_spread_cores" },
    @{ bind = "close"; places = "threads"; label = "bind_close_threads" }
)

"proc_bind,places,label,task_b_seconds,l1_miss_pct,llc_miss_pct,perf_used" | Out-File $Csv -Encoding utf8

$hasPerf = $null -ne (Get-Command perf -ErrorAction SilentlyContinue)

foreach ($cfg in $configs) {
    Remove-Item Env:OMP_PROC_BIND -ErrorAction SilentlyContinue
    Remove-Item Env:OMP_PLACES -ErrorAction SilentlyContinue
    if ($cfg.bind) { $env:OMP_PROC_BIND = $cfg.bind }
    if ($cfg.places) { $env:OMP_PLACES = $cfg.places }

    Write-Host "`n--- $($cfg.label)  OMP_PROC_BIND=$($cfg.bind)  OMP_PLACES=$($cfg.places) ---"

    $sec = 0.0
    $l1pct = ""
    $llcpct = ""
    $perfUsed = "0"

    if ($hasPerf) {
        $perfOut = Join-Path $env:TEMP "perf_affinity_$($cfg.label).txt"
        $perfEvents = "cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses"
        & perf stat -e $perfEvents -o $perfOut -- $Exe --benchmark-affinity 2>&1 | Tee-Object -Append $Log
        $line = Get-Content $perfOut | Select-String "seconds time elapsed" | ForEach-Object { $_.Line }
        if ($line -match "([\d.]+)\s+seconds") { $sec = [double]$Matches[1] }
        $stats = Get-Content $perfOut
        $l1loads = [double](($stats | Select-String "L1-dcache-loads\s+" | ForEach-Object {
            if ($_ -match "([\d,]+)\s+L1") { ($Matches[1] -replace ',','') }
        }) | Select-Object -First 1)
        $l1miss = [double](($stats | Select-String "L1-dcache-load-misses\s+" | ForEach-Object {
            if ($_ -match "([\d,]+)\s+L1") { ($Matches[1] -replace ',','') }
        }) | Select-Object -First 1)
        $cref = [double](($stats | Select-String "cache-references\s+" | ForEach-Object {
            if ($_ -match "([\d,]+)\s+cache-references") { ($Matches[1] -replace ',','') }
        }) | Select-Object -First 1)
        $cmiss = [double](($stats | Select-String "cache-misses\s+" | ForEach-Object {
            if ($_ -match "([\d,]+)\s+cache-misses") { ($Matches[1] -replace ',','') }
        }) | Select-Object -First 1)
        if ($l1loads -gt 0) { $l1pct = "{0:F2}" -f (100.0 * $l1miss / $l1loads) }
        if ($cref -gt 0) { $llcpct = "{0:F2}" -f (100.0 * $cmiss / $cref) }
        $perfUsed = "1"
    } else {
        $out = & $Exe --benchmark-affinity 2>&1 | Tee-Object -Append $Log
        $m = $out | Select-String "Task B median time:\s+([\d.]+)"
        if ($m) { $sec = [double]$m.Matches[0].Groups[1].Value }
    }

    if ($sec -eq 0.0) {
        $single = Join-Path $Root "benchmark_affinity_single.csv"
        if (Test-Path $single) {
            $last = Import-Csv $single | Select-Object -Last 1
            $sec = [double]$last.PSObject.Properties[2].Value
            Remove-Item $single -Force
        }
    }

    "$($cfg.bind),$($cfg.places),$($cfg.label),$sec,$l1pct,$llcpct,$perfUsed" | Add-Content $Csv
    Write-Host "Task B: $sec s  L1-miss%=$l1pct  LLC-miss%=$llcpct"
}

Write-Host "`nWrote $Csv"
if (-not $hasPerf) {
    Write-Host "perf not found — timing only. Install perf (WSL/Linux) for L1/LLC counters."
}
