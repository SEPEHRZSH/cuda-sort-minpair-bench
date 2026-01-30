# CUDA Benchmark: Shared vs Global Memory

A high-performance benchmarking suite designed to analyze memory latency and throughput differences between **Block-Shared Memory** and **Global Distributed Memory** on NVIDIA GPUs.

### 📋 Overview
This repository implements two distinct algorithms to stress-test memory access patterns:

- **Part A (Sorting):** A Bitonic-style parallel sorter processing integers *(Evens ascending / Odds descending)*.
- **Part B (Search):** A closest-pair algorithm calculating minimum absolute differences in large datasets.

## 🗂️ Project Architecture
The codebase is organized as follows:

```text
.
├── data/                  # Synthetic datasets and generation scripts
├── include/               # CUDA kernels (.cuh) and shared headers
├── src/                   # Host application and device logic
├── bin/                   # Compiled binaries (generated)
├── outputs/               # Performance logs and analysis artifacts
├── generate_dataset.py    # Python utility for reproducible data generation
└── plot_results_stats.py  # Statistical analysis and visualization tool
```

## 🛠️ Build Environment

### Prerequisites
- **Compiler:** MSVC (Visual Studio 2019/2022 Build Tools)
- **SDK:** NVIDIA CUDA Toolkit (v11.0 or higher)
- **Analysis:** Python 3.x (with matplotlib, pandas)

## Build (Windows / MSVC)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist bin mkdir bin

nvcc -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -DNDEBUG ^
  -Iinclude -I"%CUDA_PATH%\include" ^
  src\main.cu src\partA.cu src\partB.cu src\utils.cu src\io.cpp ^
  -Xcompiler "/EHsc /W4" ^
  -o bin\cuda_mem_bench.exe
```

## ⚡ Execution & Usage

### 1. Data Synthesis
Generate a deterministic dataset to ensure benchmark reproducibility.

```bat
python generate_dataset.py --n 1000000 --out data/dataset.txt --seed 123
```

### 2. Running Benchmarks
Execute the binary to run kernels and log performance metrics.

```bat
bin\cuda_mem_bench.exe --input data/dataset.txt --outdir outputs --mode all --variant both --repeats 10000
```

### ⚙️ Command Line Interface (CLI)

## 📊 Analytics & Visualization
Post-process the raw timing logs to generate statistical reports and graphs.

**Installation:**
```bat
pip install matplotlib pandas
```

**Generate Report:**
```bat
python plot_results_stats.py --log outputs/log.txt --out outputs/plots --title-prefix "Memory Latency Analysis"
```

**Output Artifacts:**
- `timings.csv`: Raw execution times (CSV format).
- `analysis_report.txt`: Statistical summary (Mean, StdDev, Min/Max).
- `benchmark_*.png`: Comparative bar charts and violin plots.
