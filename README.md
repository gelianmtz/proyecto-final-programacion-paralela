# Fractal + convolution (sequential baseline)

Purely sequential C++ program for a parallel-programming course baseline:

1. **Task A** — Render the Mandelbrot set at **8K UHD** (7680×4320).
2. **Task B** — Apply a **heavy 2D convolution** to the image:
   - **Gaussian** (default): full naive 2D kernel, radius 25 (51×51).
   - **Sobel**: edge map on luminance via 3×3 kernels.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

On Windows with Visual Studio:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\fractal_convolve_sequential.exe
```

## Run

```bash
./build/fractal_convolve_sequential          # Gaussian blur
./build/fractal_convolve_sequential sobel    # Sobel edges
```

Outputs (binary PPM):

- `mandelbrot_8k.ppm` — raw fractal
- `mandelbrot_8k_blur.ppm` or `mandelbrot_8k_sobel.ppm` — filtered result

View PPM files with ImageMagick, GIMP, IrfanView, or:

```bash
magick mandelbrot_8k.ppm mandelbrot_8k.png
```

## Performance note

At 8K with a 51×51 Gaussian, Task B is intentionally expensive (billions of multiply-adds). Expect long runtimes on a single core; that is suitable as a **sequential baseline** before parallelizing.

To experiment faster during development, temporarily lower `kWidth` / `kHeight` or `kBlurRadius` in `src/main.cpp`.
