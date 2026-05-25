# Build with vectorization reports (-fopt-info-vec*)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Root "src\main.cpp"
$Out = Join-Path $Root "build\fractal_convolve_parallel.exe"
$Log = Join-Path $Root "build\vectorization_report.log"

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$flags = @(
    "-std=c++17", "-O3", "-march=native", "-fopenmp",
    "-Wall", "-Wextra",
    "-ftree-vectorize", "-fopt-info-vec-optimized", "-fopt-info-vec-missed"
)

Write-Host "g++ $($flags -join ' ') -> $Out"
& g++ @flags $Src -o $Out 2>&1 | Tee-Object $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Vectorized loops (grep) ==="
Select-String -Path $Log -Pattern "vectorized|loop vectorized" | Select-Object -First 40

Write-Host "`n=== Missed vectorization in main.cpp (grep) ==="
Select-String -Path $Log -Pattern "main.cpp.*missed" | Select-Object -First 20

Write-Host "`nFull log: $Log"
