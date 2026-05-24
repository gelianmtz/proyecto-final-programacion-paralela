// OpenMP parallel baseline:
//   Task A — render Mandelbrot set at 8K resolution
//   Task B — apply a heavy 2D convolution (Gaussian blur or Sobel)

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 7680;
constexpr int kHeight = 4320;

constexpr int kMaxIterations = 750;
constexpr double kEscapeRadiusSq = 4.0;

constexpr double kCenterRe = -0.5;
constexpr double kCenterIm = 0.0;
constexpr double kZoom = 1.0;

constexpr int kBenchmarkReps = 3;
constexpr int kBenchmarkWarmup = 1;

enum class FilterKind { Gaussian, Sobel };
enum class ScheduleKind { Static, Dynamic, Guided };

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;

    Image() = default;
    Image(int w, int h) : width(w), height(h), rgb(static_cast<size_t>(w) * h * 3, 0) {}

    uint8_t* pixel(int x, int y) { return &rgb[(static_cast<size_t>(y) * width + x) * 3]; }
    const uint8_t* pixel(int x, int y) const {
        return &rgb[(static_cast<size_t>(y) * width + x) * 3];
    }
};

inline int clamp_int(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

inline double sample_channel(const Image& img, int c, int x, int y) {
    x = clamp_int(x, 0, img.width - 1);
    y = clamp_int(y, 0, img.height - 1);
    return static_cast<double>(img.pixel(x, y)[c]);
}

uint8_t to_byte(double v) {
    v = std::max(0.0, std::min(255.0, std::round(v)));
    return static_cast<uint8_t>(v);
}

void colorize(int iterations, double smooth_t, uint8_t* out_rgb) {
    if (iterations >= kMaxIterations) {
        out_rgb[0] = out_rgb[1] = out_rgb[2] = 0;
        return;
    }

    const double t = smooth_t + static_cast<double>(iterations);
    const double hue = std::fmod(0.03 * t + 0.65, 1.0);
    const double sat = 0.85;
    const double val = std::min(1.0, 0.15 + 0.85 * (static_cast<double>(iterations) / kMaxIterations));

    const int hi = static_cast<int>(hue * 6.0) % 6;
    const double f = hue * 6.0 - std::floor(hue * 6.0);
    const double p = val * (1.0 - sat);
    const double q = val * (1.0 - f * sat);
    const double tcol = val * (1.0 - (1.0 - f) * sat);

    double r = 0, g = 0, b = 0;
    switch (hi) {
        case 0: r = val; g = tcol; b = p; break;
        case 1: r = q; g = val; b = p; break;
        case 2: r = p; g = val; b = tcol; break;
        case 3: r = p; g = q; b = val; break;
        case 4: r = tcol; g = p; b = val; break;
        default: r = val; g = p; b = q; break;
    }

    out_rgb[0] = to_byte(r * 255.0);
    out_rgb[1] = to_byte(g * 255.0);
    out_rgb[2] = to_byte(b * 255.0);
}

void render_mandelbrot_row(Image& img, int y, double half_w, double half_h) {
    const double c_im = kCenterIm + (static_cast<double>(y) / (img.height - 1) * 2.0 - 1.0) * half_h;
    for (int x = 0; x < img.width; ++x) {
        const double c_re =
            kCenterRe + (static_cast<double>(x) / (img.width - 1) * 2.0 - 1.0) * half_w;

        double z_re = 0.0;
        double z_im = 0.0;
        int iter = 0;

        while (iter < kMaxIterations) {
            const double z_re2 = z_re * z_re;
            const double z_im2 = z_im * z_im;
            if (z_re2 + z_im2 > kEscapeRadiusSq) {
                break;
            }
            z_im = 2.0 * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
            ++iter;
        }

        double smooth_t = static_cast<double>(iter);
        if (iter < kMaxIterations) {
            const double mag2 = z_re * z_re + z_im * z_im;
            smooth_t += 1.0 - std::log2(std::log(mag2) / std::log(2.0));
        }

        colorize(iter, smooth_t, img.pixel(x, y));
    }
}

void render_mandelbrot_static_default(Image& img) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double half_w = 1.5 * kZoom * aspect;
    const double half_h = 1.5 * kZoom;

#pragma omp parallel for schedule(static)
    for (int y = 0; y < img.height; ++y) {
        render_mandelbrot_row(img, y, half_w, half_h);
    }
}

void render_mandelbrot_static(Image& img, int chunk) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double half_w = 1.5 * kZoom * aspect;
    const double half_h = 1.5 * kZoom;

#pragma omp parallel for schedule(static, chunk)
    for (int y = 0; y < img.height; ++y) {
        render_mandelbrot_row(img, y, half_w, half_h);
    }
}

void render_mandelbrot_dynamic(Image& img, int chunk) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double half_w = 1.5 * kZoom * aspect;
    const double half_h = 1.5 * kZoom;

#pragma omp parallel for schedule(dynamic, chunk)
    for (int y = 0; y < img.height; ++y) {
        render_mandelbrot_row(img, y, half_w, half_h);
    }
}

void render_mandelbrot_guided(Image& img, int chunk) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double half_w = 1.5 * kZoom * aspect;
    const double half_h = 1.5 * kZoom;

#pragma omp parallel for schedule(guided, chunk)
    for (int y = 0; y < img.height; ++y) {
        render_mandelbrot_row(img, y, half_w, half_h);
    }
}

void render_mandelbrot(Image& img, ScheduleKind kind, int chunk) {
    switch (kind) {
        case ScheduleKind::Static:
            render_mandelbrot_static(img, chunk);
            break;
        case ScheduleKind::Dynamic:
            render_mandelbrot_dynamic(img, chunk);
            break;
        case ScheduleKind::Guided:
            render_mandelbrot_guided(img, chunk);
            break;
    }
}

const char* schedule_name(ScheduleKind kind) {
    switch (kind) {
        case ScheduleKind::Static:
            return "static";
        case ScheduleKind::Dynamic:
            return "dynamic";
        case ScheduleKind::Guided:
            return "guided";
    }
    return "unknown";
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    return v[mid];
}

double time_task_a(ScheduleKind kind, int chunk) {
    Image img(kWidth, kHeight);
    const auto t0 = std::chrono::steady_clock::now();
    render_mandelbrot(img, kind, chunk);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

double benchmark_task_a(ScheduleKind kind, int chunk) {
    for (int w = 0; w < kBenchmarkWarmup; ++w) {
        (void)time_task_a(kind, chunk);
    }

    std::vector<double> samples;
    samples.reserve(kBenchmarkReps);
    for (int r = 0; r < kBenchmarkReps; ++r) {
        samples.push_back(time_task_a(kind, chunk));
    }
    return median(samples);
}

struct BenchmarkRow {
    ScheduleKind kind;
    int chunk = 0;
    double seconds = 0.0;
};

double benchmark_static_default() {
    for (int w = 0; w < kBenchmarkWarmup; ++w) {
        Image img(kWidth, kHeight);
        render_mandelbrot_static_default(img);
    }
    std::vector<double> samples;
    samples.reserve(kBenchmarkReps);
    for (int r = 0; r < kBenchmarkReps; ++r) {
        Image img(kWidth, kHeight);
        const auto t0 = std::chrono::steady_clock::now();
        render_mandelbrot_static_default(img);
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    return median(samples);
}

void run_scheduler_benchmark() {
    const std::vector<int> chunk_sizes = {1,    2,    4,    8,    16,   32,   64,
                                          128,  256,  512,  1024, 2048, 4320};
    std::vector<BenchmarkRow> results;

    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "Resolution: " << kWidth << " x " << kHeight << "\n";
    std::cout << "Task A benchmark: " << kBenchmarkWarmup << " warmup + " << kBenchmarkReps
              << " timed runs (median reported)\n\n";

    const double static_default_sec = benchmark_static_default();
    std::cout << "OpenMP default static (no chunk): " << std::fixed << std::setprecision(3)
              << static_default_sec << " s\n\n";

    for (ScheduleKind kind :
         {ScheduleKind::Static, ScheduleKind::Dynamic, ScheduleKind::Guided}) {
        std::cout << "Testing schedule(" << schedule_name(kind) << ") ...\n";
        for (int chunk : chunk_sizes) {
            const double sec = benchmark_task_a(kind, chunk);
            results.push_back({kind, chunk, sec});
            std::cout << "  chunk=" << std::setw(5) << chunk << "  " << std::fixed
                      << std::setprecision(3) << sec << " s\n";
        }
        std::cout << "\n";
    }

    std::sort(results.begin(), results.end(),
              [](const BenchmarkRow& a, const BenchmarkRow& b) { return a.seconds < b.seconds; });

    const BenchmarkRow& best = results.front();

    std::ofstream csv("benchmark_task_a.csv");
    csv << "schedule,chunk,seconds\n";
    csv << "static_default,0," << std::setprecision(6) << static_default_sec << '\n';
    for (const BenchmarkRow& row : results) {
        csv << schedule_name(row.kind) << ',' << row.chunk << ',' << std::setprecision(6)
            << row.seconds << '\n';
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Summary ===\n";
    std::cout << "Static default (schedule(static), implicit chunk): " << static_default_sec
              << " s\n";
    std::cout << "Best explicit: schedule(" << schedule_name(best.kind) << "), chunk="
              << best.chunk << ", " << best.seconds << " s\n";

    if (static_default_sec > 0.0) {
        const double speedup = static_default_sec / best.seconds;
        std::cout << "Speedup vs static default: " << std::setprecision(2) << speedup << "x\n";
    }

    std::cout << "\nTop 5 configurations:\n";
    std::cout << std::setw(10) << "schedule" << std::setw(8) << "chunk" << std::setw(12)
              << "time (s)\n";
    for (size_t i = 0; i < std::min<size_t>(5, results.size()); ++i) {
        const BenchmarkRow& row = results[i];
        std::cout << std::setw(10) << schedule_name(row.kind) << std::setw(8) << row.chunk
                  << std::setw(12) << row.seconds << "\n";
    }
    std::cout << "\nFull results: benchmark_task_a.csv\n";
}

// --- Task B: convolution ------------------------------------------------------

std::vector<double> make_gaussian_kernel(int radius, double sigma) {
    const int size = 2 * radius + 1;
    std::vector<double> kernel(size, 0.0);
    const double denom = 2.0 * sigma * sigma;
    double sum = 0.0;

    for (int i = 0; i < size; ++i) {
        const int d = i - radius;
        kernel[i] = std::exp(-(static_cast<double>(d * d)) / denom);
        sum += kernel[i];
    }
    for (double& v : kernel) {
        v /= sum;
    }
    return kernel;
}

Image convolve_2d_gaussian(const Image& src, int radius, double sigma) {
    const std::vector<double> kernel = make_gaussian_kernel(radius, sigma);
    const int ksize = static_cast<int>(kernel.size());
    Image dst(src.width, src.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (int ky = 0; ky < ksize; ++ky) {
                    const int sy = y + ky - radius;
                    const double wy = kernel[ky];
                    for (int kx = 0; kx < ksize; ++kx) {
                        const int sx = x + kx - radius;
                        acc += wy * kernel[kx] * sample_channel(src, c, sx, sy);
                    }
                }
                dst.pixel(x, y)[c] = to_byte(acc);
            }
        }
    }
    return dst;
}

Image convolve_sobel(const Image& src) {
    static const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    Image dst(src.width, src.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double lum_x = 0.0;
            double lum_y = 0.0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* p = src.pixel(x + kx, y + ky);
                    const double lum = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
                    lum_x += gx[ky + 1][kx + 1] * lum;
                    lum_y += gy[ky + 1][kx + 1] * lum;
                }
            }

            const double mag = std::sqrt(lum_x * lum_x + lum_y * lum_y);
            const uint8_t v = to_byte(mag);
            uint8_t* out = dst.pixel(x, y);
            out[0] = out[1] = out[2] = v;
        }
    }
    return dst;
}

bool write_ppm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.rgb.data()),
              static_cast<std::streamsize>(img.rgb.size()));
    return static_cast<bool>(out);
}

FilterKind parse_filter(const std::string& arg) {
    if (arg == "sobel" || arg == "Sobel") {
        return FilterKind::Sobel;
    }
    return FilterKind::Gaussian;
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " [gaussian|sobel]\n"
              << "  " << prog << " --benchmark-a\n"
              << "\n"
              << "  gaussian (default) — wide 2D Gaussian blur (radius 25, sigma 8)\n"
              << "  sobel              — Sobel edge map on luminance\n"
              << "  --benchmark-a      — sweep static/dynamic/guided chunk sizes for Task A\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    FilterKind filter = FilterKind::Gaussian;
    bool benchmark_a = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--benchmark-a" || arg == "benchmark-a") {
            benchmark_a = true;
            continue;
        }
        filter = parse_filter(arg);
    }

    if (benchmark_a) {
        run_scheduler_benchmark();
        return 0;
    }

    // Optimal on this machine (16 threads, --benchmark-a): static, chunk=16.
    constexpr ScheduleKind kTaskASchedule = ScheduleKind::Static;
    constexpr int kTaskAChunk = 16;

    std::cout << "Resolution: " << kWidth << " x " << kHeight << " (8K UHD)\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "Task A schedule: " << schedule_name(kTaskASchedule) << ", chunk="
              << kTaskAChunk << "\n\n";

    Image fractal(kWidth, kHeight);

    const auto t0 = std::chrono::steady_clock::now();
    render_mandelbrot(fractal, kTaskASchedule, kTaskAChunk);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec_a = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Task A (Mandelbrot): " << sec_a << " s\n";

    if (!write_ppm("mandelbrot_8k.ppm", fractal)) {
        std::cerr << "Failed to write mandelbrot_8k.ppm\n";
        return 1;
    }
    std::cout << "Wrote mandelbrot_8k.ppm\n";

    const auto t2 = std::chrono::steady_clock::now();

    Image filtered;
    if (filter == FilterKind::Gaussian) {
        constexpr int kBlurRadius = 25;
        constexpr double kBlurSigma = 8.0;
        std::cout << "Task B: 2D Gaussian convolution (radius " << kBlurRadius
                  << ", sigma " << kBlurSigma << ")\n";
        filtered = convolve_2d_gaussian(fractal, kBlurRadius, kBlurSigma);
    } else {
        std::cout << "Task B: Sobel edge detection (3x3 kernels)\n";
        filtered = convolve_sobel(fractal);
    }

    const auto t3 = std::chrono::steady_clock::now();
    const double sec_b = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "Task B (convolution): " << sec_b << " s\n";

    const std::string out_name =
        (filter == FilterKind::Gaussian) ? "mandelbrot_8k_blur.ppm" : "mandelbrot_8k_sobel.ppm";
    if (!write_ppm(out_name, filtered)) {
        std::cerr << "Failed to write " << out_name << "\n";
        return 1;
    }
    std::cout << "Wrote " << out_name << "\n";
    std::cout << "Total: " << (sec_a + sec_b) << " s\n";

    return 0;
}
