#include "partA.h"
#include "cuda_check.h"
#include "device_buffer.h"
#include "gpu_timer.h"
#include "io.h"
#include "utils.h"
#include "config.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

namespace hw3 {

static __global__ void partition_atomic_kernel(const i32* __restrict__ in, u32 n,
                                               i32* __restrict__ evens, i32* __restrict__ odds,
                                               u32* __restrict__ even_count, u32* __restrict__ odd_count) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  i32 x = in[tid];
  if ((x & 1) == 0) {
    u32 pos = atomicAdd(even_count, 1u);
    evens[pos] = x;
  } else {
    u32 pos = atomicAdd(odd_count, 1u);
    odds[pos] = x;
  }
}

static __global__ void reverse_inplace(i32* data, u32 n) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  u32 j = n - 1 - i;
  if (i >= n/2) return;
  i32 tmp = data[i];
  data[i] = data[j];
  data[j] = tmp;
}

static __global__ void concat_kernel(const i32* evens, u32 ne,
                                     const i32* odds,  u32 no,
                                     i32* out) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  u32 n = ne + no;
  if (tid >= n) return;
  out[tid] = (tid < ne) ? evens[tid] : odds[tid - ne];
}

struct BlockCounts { u32 even; u32 odd; };

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void count_blocks_kernel(const i32* __restrict__ in, u32 n, BlockCounts* __restrict__ counts) {
  using BlockReduce = cub::BlockReduce<u32, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage tmp;

  const u32 block_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  u32 block_start = blockIdx.x * block_items;

  u32 local_even = 0;
  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    u32 idx = block_start + threadIdx.x + it * BLOCK_THREADS;
    if (idx < n) {
      i32 x = in[idx];
      local_even += ((x & 1) == 0) ? 1u : 0u;
    }
  }

  u32 block_even = BlockReduce(tmp).Sum(local_even);
  __syncthreads();

  if (threadIdx.x == 0) {
    u32 end = block_start + block_items;
    u32 items_in_block = (end <= n) ? block_items : (n > block_start ? (n - block_start) : 0u);
    u32 block_odd = items_in_block - block_even;
    counts[blockIdx.x] = BlockCounts{block_even, block_odd};
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void scatter_blocks_kernel(const i32* __restrict__ in, u32 n,
                                      const u32* __restrict__ even_offsets,
                                      const u32* __restrict__ odd_offsets,
                                      i32* __restrict__ evens,
                                      i32* __restrict__ odds) {
  using BlockScan = cub::BlockScan<u32, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage tmp_even;
  __shared__ typename BlockScan::TempStorage tmp_odd;
  __shared__ u32 lane_prefix_even[ITEMS_PER_THREAD];
  __shared__ u32 lane_prefix_odd[ITEMS_PER_THREAD];

  const u32 block_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  u32 block_start = blockIdx.x * block_items;

  u32 base_even = even_offsets[blockIdx.x];
  u32 base_odd  = odd_offsets[blockIdx.x];

  u32 cum_even = 0;
  u32 cum_odd  = 0;

  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    u32 idx = block_start + threadIdx.x + it * BLOCK_THREADS;

    u32 is_even = 0, is_odd = 0;
    i32 x = 0;
    if (idx < n) {
      x = in[idx];
      is_even = ((x & 1) == 0) ? 1u : 0u;
      is_odd = 1u - is_even;
    }

    u32 even_prefix = 0, odd_prefix = 0;
    u32 even_total = 0, odd_total = 0;

    BlockScan(tmp_even).ExclusiveSum(is_even, even_prefix, even_total);
    __syncthreads();
    BlockScan(tmp_odd).ExclusiveSum(is_odd, odd_prefix, odd_total);
    __syncthreads();

    if (threadIdx.x == 0) {
      lane_prefix_even[it] = cum_even;
      lane_prefix_odd[it]  = cum_odd;
      cum_even += even_total;
      cum_odd  += odd_total;
    }
    __syncthreads();

    if (idx < n) {
      if (is_even) evens[base_even + lane_prefix_even[it] + even_prefix] = x;
      else         odds [base_odd  + lane_prefix_odd[it]  + odd_prefix ] = x;
    }
    __syncthreads();
  }
}

static void exclusive_scan_u32(std::vector<u32>& v) {
  u32 acc = 0;
  for (auto& x : v) { u32 t = x; x = acc; acc += t; }
}

static void ensure_temp(void** d_temp, std::size_t& cap_bytes, std::size_t need_bytes) {
  if (need_bytes <= cap_bytes) return;
  if (*d_temp) { cudaFree(*d_temp); *d_temp = nullptr; }
  CUDA_CHECK(cudaMalloc(d_temp, need_bytes));
  cap_bytes = need_bytes;
}

static PartAResult run_partA_impl(const std::vector<i32>& input, bool use_shared_partition,
                                  bool save_files, const std::string& outdir, int repeats,
                                  const std::string& tag) {
  ensure_dir(outdir);

  const u32 n = static_cast<u32>(input.size());
  PartAResult res;
  if (n == 0) return res;

  DeviceBuffer<i32> d_in(n);
  CUDA_CHECK(cudaMemcpy(d_in.data(), input.data(), n * sizeof(i32), cudaMemcpyHostToDevice));

  // buffers for partitioned data
  DeviceBuffer<i32> d_evens0(n), d_evens1(n);
  DeviceBuffer<i32> d_odds0(n),  d_odds1(n);
  DeviceBuffer<i32> d_out(n);

  // Counters for atomic version
  DeviceBuffer<u32> d_even_count(1), d_odd_count(1);

  void* d_temp = nullptr;
  std::size_t temp_cap = 0;

  auto do_one = [&]() -> float {
    u32 even_count = 0, odd_count = 0;

    if (!use_shared_partition) {
      CUDA_CHECK(cudaMemset(d_even_count.data(), 0, sizeof(u32)));
      CUDA_CHECK(cudaMemset(d_odd_count.data(), 0, sizeof(u32)));
    }

    GpuTimer timer;
    timer.start();

    if (!use_shared_partition) {
      int threads = 256;
      int blocks = (n + threads - 1) / threads;
      partition_atomic_kernel<<<blocks, threads>>>(d_in.data(), n, d_evens0.data(), d_odds0.data(),
                                                   d_even_count.data(), d_odd_count.data());
      CUDA_SYNC_CHECK();
      CUDA_CHECK(cudaMemcpy(&even_count, d_even_count.data(), sizeof(u32), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&odd_count,  d_odd_count.data(),  sizeof(u32), cudaMemcpyDeviceToHost));
    } else {
      constexpr int BT = PARTA_BLOCK_THREADS;
      constexpr int IPT = PARTA_ITEMS_PER_THREAD;
      const u32 block_items = BT * IPT;
      const u32 num_blocks = (n + block_items - 1) / block_items;

      DeviceBuffer<BlockCounts> d_counts(num_blocks);
      count_blocks_kernel<BT, IPT><<<num_blocks, BT>>>(d_in.data(), n, d_counts.data());
      CUDA_SYNC_CHECK();

      std::vector<BlockCounts> h_counts(num_blocks);
      CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts.data(), num_blocks * sizeof(BlockCounts),
                            cudaMemcpyDeviceToHost));

      std::vector<u32> even_offsets(num_blocks), odd_offsets(num_blocks);
      for (u32 i = 0; i < num_blocks; ++i) {
        even_offsets[i] = h_counts[i].even;
        odd_offsets[i]  = h_counts[i].odd;
      }

      exclusive_scan_u32(even_offsets);
      exclusive_scan_u32(odd_offsets);

      if (num_blocks) {
        even_count = even_offsets.back() + h_counts.back().even;
        odd_count  = odd_offsets.back()  + h_counts.back().odd;
      }

      DeviceBuffer<u32> d_even_offsets(num_blocks), d_odd_offsets(num_blocks);
      CUDA_CHECK(cudaMemcpy(d_even_offsets.data(), even_offsets.data(), num_blocks * sizeof(u32), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_odd_offsets.data(),  odd_offsets.data(),  num_blocks * sizeof(u32), cudaMemcpyHostToDevice));

      scatter_blocks_kernel<BT, IPT><<<num_blocks, BT>>>(d_in.data(), n,
                                                         d_even_offsets.data(), d_odd_offsets.data(),
                                                         d_evens0.data(), d_odds0.data());
      CUDA_SYNC_CHECK();
    }

    res.even_count = even_count;
    res.odd_count  = odd_count;

    // ---- Sort evens ascending (CUB DoubleBuffer) ----
    cub::DoubleBuffer<i32> evens_buf(d_evens0.data(), d_evens1.data());
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, evens_buf, even_count));
    ensure_temp(&d_temp, temp_cap, temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, evens_buf, even_count));
    CUDA_SYNC_CHECK();

    // ---- Sort odds ascending then reverse => descending ----
    cub::DoubleBuffer<i32> odds_buf(d_odds0.data(), d_odds1.data());
    temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, odds_buf, odd_count));
    ensure_temp(&d_temp, temp_cap, temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, odds_buf, odd_count));
    CUDA_SYNC_CHECK();

    i32* odds_sorted = odds_buf.Current();
    if (odd_count > 1) {
      int threads = 256;
      int blocks = ((odd_count / 2) + threads - 1) / threads;
      reverse_inplace<<<blocks, threads>>>(odds_sorted, odd_count);
      CUDA_SYNC_CHECK();
    }

    // ---- Concatenate ----
    const i32* evens_sorted = evens_buf.Current();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    concat_kernel<<<blocks, threads>>>(evens_sorted, even_count, odds_sorted, odd_count, d_out.data());
    CUDA_SYNC_CHECK();

    return timer.stop_ms();
  };

  (void)do_one(); // warmup

  float best = 1e30f;
  for (int r = 0; r < std::max(1, repeats); ++r) best = std::min(best, do_one());
  res.ms_total = best;

  res.output.resize(n);
  CUDA_CHECK(cudaMemcpy(res.output.data(), d_out.data(), n * sizeof(i32), cudaMemcpyDeviceToHost));

  if (d_temp) cudaFree(d_temp);

  if (save_files) {
    std::string out_path = (std::filesystem::path(outdir) / ("partA_" + tag + "_output.txt")).string();
    write_csv_line(out_path, res.output);

    std::string log_path = (std::filesystem::path(outdir) / "log.txt").string();
    std::ostringstream oss;
    oss << "=== Part A (" << tag << ") ===\n"
        << "N=" << n << " even_count=" << res.even_count << " odd_count=" << res.odd_count << "\n"
        << "best_ms=" << std::fixed << std::setprecision(3) << res.ms_total << "\n"
        << "preview=" << preview_vec(res.output, 20) << "\n"
        << "verify=" << (verify_partA(res.output, res.even_count) ? "OK" : "FAIL") << "\n\n";
    append_text(log_path, oss.str());
  }

  return res;
}

PartAResult run_partA_distributed(const std::vector<i32>& input, bool save_files,
                                 const std::string& outdir, int repeats) {
  return run_partA_impl(input, false, save_files, outdir, repeats, "distributed");
}

PartAResult run_partA_shared(const std::vector<i32>& input, bool save_files,
                             const std::string& outdir, int repeats) {
  return run_partA_impl(input, true, save_files, outdir, repeats, "shared");
}

} // namespace hw3
