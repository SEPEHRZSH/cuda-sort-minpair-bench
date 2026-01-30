#include "partA.h"
#include "partB.h"
#include "io.h"
#include "utils.h"
#include "cuda_check.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace hw3 {

struct Args {
  std::string input = "data/dataset.txt";
  std::string outdir = "outputs";
  std::string mode = "all";        // all | A | B
  std::string variant = "both";    // both | shared | distributed
  int repeats = DEFAULT_REPEATS;
  bool save = true;
};

static void print_usage() {
  std::cout <<
R"(parallel_hw3 usage:
  parallel_hw3 --input <path> --outdir <dir> [--mode all|A|B] [--variant both|shared|distributed] [--repeats N] [--nosave]

Examples:
  parallel_hw3 --input data/dataset.txt --outdir outputs --mode all --variant both --repeats 5
  parallel_hw3 --input data/dataset.txt --mode A --variant shared
)";
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(2); }
      return std::string(argv[++i]);
    };

    if (k == "--input") a.input = need("--input");
    else if (k == "--outdir") a.outdir = need("--outdir");
    else if (k == "--mode") a.mode = need("--mode");
    else if (k == "--variant") a.variant = need("--variant");
    else if (k == "--repeats") a.repeats = std::stoi(need("--repeats"));
    else if (k == "--nosave") a.save = false;
    else if (k == "--help" || k == "-h") { print_usage(); std::exit(0); }
    else { std::cerr << "Unknown arg: " << k << "\n"; print_usage(); std::exit(2); }
  }
  if (a.repeats < 1) a.repeats = 1;
  return a;
}

static void print_device_info() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Using GPU: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")\n";
}

struct Stats {
  int repeats = 0;
  double best = 0.0;
  double avg = 0.0;
  double median = 0.0;
  double stddev = 0.0;
  double minv = 0.0;
  double maxv = 0.0;
};

static Stats compute_stats(const std::vector<double>& xs) {
  Stats s;
  s.repeats = static_cast<int>(xs.size());
  if (xs.empty()) return s;

  s.minv = *std::min_element(xs.begin(), xs.end());
  s.maxv = *std::max_element(xs.begin(), xs.end());
  s.best = s.minv;

  const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
  s.avg = sum / xs.size();

  std::vector<double> tmp = xs;
  std::sort(tmp.begin(), tmp.end());
  if (tmp.size() % 2 == 1) {
    s.median = tmp[tmp.size() / 2];
  } else {
    const std::size_t i = tmp.size() / 2;
    s.median = 0.5 * (tmp[i - 1] + tmp[i]);
  }

  double var = 0.0;
  for (double x : xs) {
    const double d = x - s.avg;
    var += d * d;
  }
  var /= xs.size();
  s.stddev = std::sqrt(var);

  return s;
}

static void truncate_file(const std::filesystem::path& p) {
  std::ofstream(p.string(), std::ios::trunc).close();
}

static void append_stats_block_partA(const std::filesystem::path& stats_path,
                                     const char* variant,
                                     std::size_t n,
                                     u32 even_count, u32 odd_count,
                                     const Stats& st,
                                     bool verify_ok) {
  std::ofstream f(stats_path.string(), std::ios::app);
  f << "=== Part A (" << variant << ") ===\n";
  f << "N=" << n << " even_count=" << even_count << " odd_count=" << odd_count << "\n";
  f << "repeats=" << st.repeats << "\n";
  f << std::fixed << std::setprecision(6);
  f << "best_ms=" << st.best << "\n";
  f << "avg_ms=" << st.avg << "\n";
  f << "median_ms=" << st.median << "\n";
  f << "std_ms=" << st.stddev << "\n";
  f << "min_ms=" << st.minv << "\n";
  f << "max_ms=" << st.maxv << "\n";
  f << "verify=" << (verify_ok ? "OK" : "FAIL") << "\n\n";
}

static void append_stats_block_partB(const std::filesystem::path& stats_path,
                                     const char* variant,
                                     std::size_t n,
                                     i32 a, i32 b, i32 diff,
                                     const Stats& st) {
  std::ofstream f(stats_path.string(), std::ios::app);
  f << "=== Part B (" << variant << ") ===\n";
  f << "N=" << n << " pair=(" << a << ", " << b << ") diff=" << diff << "\n";
  f << "repeats=" << st.repeats << "\n";
  f << std::fixed << std::setprecision(6);
  f << "best_ms=" << st.best << "\n";
  f << "avg_ms=" << st.avg << "\n";
  f << "median_ms=" << st.median << "\n";
  f << "std_ms=" << st.stddev << "\n";
  f << "min_ms=" << st.minv << "\n";
  f << "max_ms=" << st.maxv << "\n\n";
}

} 

int main(int argc, char** argv) {
  using namespace hw3;

  try {
    Args args = parse_args(argc, argv);

    ensure_dir(args.outdir);
    print_device_info();

    std::cout << "Reading input: " << args.input << "\n";
    std::vector<i32> input = read_dataset(args.input);
    std::cout << "Loaded N=" << input.size() << "\n";

    if (input.size() < 2) {
      std::cerr << "Need at least 2 numbers.\n";
      return 1;
    }

    const bool wantA = (args.mode == "all" || args.mode == "A" || args.mode == "a");
    const bool wantB = (args.mode == "all" || args.mode == "B" || args.mode == "b");

    const bool do_distributed = (args.variant == "both" || args.variant == "distributed");
    const bool do_shared = (args.variant == "both" || args.variant == "shared");

    std::filesystem::path outdir = std::filesystem::path(args.outdir);
    std::filesystem::path log_path = outdir / "log.txt";
    std::filesystem::path stats_path = outdir / "stats.txt";

    if (args.save) {
      truncate_file(log_path);
      truncate_file(stats_path);
    }

    // --------------------
    // Part A benchmarking
    // --------------------
    if (wantA) {
      std::cout << "\n[Part A] even->asc then odd->desc\n";

      if (do_distributed) {
        std::vector<double> times;
        times.reserve(args.repeats);
        decltype(run_partA_distributed(input, false, args.outdir, 1)) best_run{};
        double best = 1e100;

        for (int i = 0; i < args.repeats; ++i) {
          auto r = run_partA_distributed(input, false, args.outdir, 1);
          times.push_back(r.ms_total);
          if (r.ms_total < best) { best = r.ms_total; best_run = std::move(r); }
        }

        Stats st = compute_stats(times);
        const bool ok = verify_partA(best_run.output, best_run.even_count);

        std::cout << "  distributed: "
                  << "best_ms=" << std::fixed << std::setprecision(3) << st.best
                  << " avg_ms=" << st.avg
                  << " median_ms=" << st.median
                  << " std_ms=" << st.stddev
                  << " min_ms=" << st.minv
                  << " max_ms=" << st.maxv
                  << " (repeats=" << st.repeats << ")"
                  << " even=" << best_run.even_count << " odd=" << best_run.odd_count
                  << " verify=" << (ok ? "OK" : "FAIL")
                  << "\n  preview=" << preview_vec(best_run.output, 20) << "\n";

        if (args.save) {
          append_stats_block_partA(stats_path, "distributed", input.size(),
                                   best_run.even_count, best_run.odd_count, st, ok);
          (void)run_partA_distributed(input, true, args.outdir, 1);
        }
      }

      if (do_shared) {
        std::vector<double> times;
        times.reserve(args.repeats);
        decltype(run_partA_shared(input, false, args.outdir, 1)) best_run{};
        double best = 1e100;

        for (int i = 0; i < args.repeats; ++i) {
          auto r = run_partA_shared(input, false, args.outdir, 1);
          times.push_back(r.ms_total);
          if (r.ms_total < best) { best = r.ms_total; best_run = std::move(r); }
        }

        Stats st = compute_stats(times);
        const bool ok = verify_partA(best_run.output, best_run.even_count);

        std::cout << "  shared:      "
                  << "best_ms=" << std::fixed << std::setprecision(3) << st.best
                  << " avg_ms=" << st.avg
                  << " median_ms=" << st.median
                  << " std_ms=" << st.stddev
                  << " min_ms=" << st.minv
                  << " max_ms=" << st.maxv
                  << " (repeats=" << st.repeats << ")"
                  << " even=" << best_run.even_count << " odd=" << best_run.odd_count
                  << " verify=" << (ok ? "OK" : "FAIL")
                  << "\n  preview=" << preview_vec(best_run.output, 20) << "\n";

        if (args.save) {
          append_stats_block_partA(stats_path, "shared", input.size(),
                                   best_run.even_count, best_run.odd_count, st, ok);
          (void)run_partA_shared(input, true, args.outdir, 1);
        }
      }
    }

    // --------------------
    // Part B benchmarking
    // --------------------
    if (wantB) {
      std::cout << "\n[Part B] closest pair (min absolute difference)\n";

      if (do_distributed) {
        std::vector<double> times;
        times.reserve(args.repeats);
        decltype(run_partB_distributed(input, false, args.outdir, 1)) best_run{};
        double best = 1e100;

        for (int i = 0; i < args.repeats; ++i) {
          auto r = run_partB_distributed(input, false, args.outdir, 1);
          times.push_back(r.ms_total);
          if (r.ms_total < best) { best = r.ms_total; best_run = std::move(r); }
        }

        Stats st = compute_stats(times);

        std::cout << "  distributed: "
                  << "best_ms=" << std::fixed << std::setprecision(3) << st.best
                  << " avg_ms=" << st.avg
                  << " median_ms=" << st.median
                  << " std_ms=" << st.stddev
                  << " min_ms=" << st.minv
                  << " max_ms=" << st.maxv
                  << " (repeats=" << st.repeats << ")"
                  << " pair=(" << best_run.a << ", " << best_run.b << ") diff=" << best_run.diff << "\n";

        if (args.save) {
          append_stats_block_partB(stats_path, "distributed", input.size(),
                                   best_run.a, best_run.b, best_run.diff, st);
          (void)run_partB_distributed(input, true, args.outdir, 1);
        }
      }

      if (do_shared) {
        std::vector<double> times;
        times.reserve(args.repeats);
        decltype(run_partB_shared(input, false, args.outdir, 1)) best_run{};
        double best = 1e100;

        for (int i = 0; i < args.repeats; ++i) {
          auto r = run_partB_shared(input, false, args.outdir, 1);
          times.push_back(r.ms_total);
          if (r.ms_total < best) { best = r.ms_total; best_run = std::move(r); }
        }

        Stats st = compute_stats(times);

        std::cout << "  shared:      "
                  << "best_ms=" << std::fixed << std::setprecision(3) << st.best
                  << " avg_ms=" << st.avg
                  << " median_ms=" << st.median
                  << " std_ms=" << st.stddev
                  << " min_ms=" << st.minv
                  << " max_ms=" << st.maxv
                  << " (repeats=" << st.repeats << ")"
                  << " pair=(" << best_run.a << ", " << best_run.b << ") diff=" << best_run.diff << "\n";

        if (args.save) {
          append_stats_block_partB(stats_path, "shared", input.size(),
                                   best_run.a, best_run.b, best_run.diff, st);
          (void)run_partB_shared(input, true, args.outdir, 1);
        }
      }
    }

    if (args.save) {
      std::cout << "\nStats written to: " << stats_path.string() << "\n";
      std::cout << "Legacy log (best-only, per last saved run) is in: " << log_path.string() << "\n";
    }

    std::cout << "\nDone. Outputs (if enabled) are in: " << args.outdir << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nERROR: " << e.what() << "\n";
    return 1;
  }
}
