#include "partB.h"
#include "cuda_check.h"
#include "device_buffer.h"
#include "gpu_timer.h"
#include "io.h"
#include "config.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <limits>
#include <algorithm>
#include <vector>

namespace hw3 {

static __device__ inline u64 atomicMin_u64(u64* addr, u64 val) {
  u64 old = *addr;
  while (val < old) {
    u64 assumed = old;
    old = atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                    static_cast<unsigned long long>(assumed),
                    static_cast<unsigned long long>(val));
    if (old == assumed) break;
  }
  return old;
}

static __host__ __device__ inline u64 pack_key(u32 diff, u32 idx) {
  return (static_cast<u64>(diff) << 32) | static_cast<u64>(idx);
}
static __host__ inline void unpack_key(u64 key, u32& diff, u32& idx) {
  diff = static_cast<u32>(key >> 32);
  idx  = static_cast<u32>(key & 0xFFFFFFFFu);
}

static __global__ void minpair_atomic_kernel(const i32* sorted, u32 n, u64* best_key) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  u32 total = gridDim.x * blockDim.x;
  for (u32 i = tid; i + 1 < n; i += total) {
    long long a = (long long)sorted[i];
    long long b = (long long)sorted[i + 1];
    u32 diff = (u32)llabs(b - a);
    atomicMin_u64(best_key, pack_key(diff, i));
  }
}

static __global__ void minpair_blockreduce_kernel(const i32* sorted, u32 n, u64* block_out) {
  __shared__ u64 sdata[PARTB_BLOCK_THREADS];
  u32 tid = threadIdx.x;
  u32 gtid = blockIdx.x * blockDim.x + tid;
  u32 stride = gridDim.x * blockDim.x;

  u64 best = pack_key(0xFFFFFFFFu, 0u);
  for (u32 i = gtid; i + 1 < n; i += stride) {
    long long a = (long long)sorted[i];
    long long b = (long long)sorted[i + 1];
    u32 diff = (u32)llabs(b - a);
    u64 key = pack_key(diff, i);
    if (key < best) best = key;
  }

  sdata[tid] = best;
  __syncthreads();

  for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      u64 other = sdata[tid + s];
      if (other < sdata[tid]) sdata[tid] = other;
    }
    __syncthreads();
  }
  if (tid == 0) block_out[blockIdx.x] = sdata[0];
}

static __global__ void reduce_keys_kernel(const u64* in, u32 n, u64* out) {
  __shared__ u64 sdata[PARTB_BLOCK_THREADS];
  u32 tid = threadIdx.x;
  u32 i = blockIdx.x * blockDim.x + tid;

  u64 v = pack_key(0xFFFFFFFFu, 0u);
  if (i < n) v = in[i];
  sdata[tid] = v;
  __syncthreads();

  for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      u64 other = sdata[tid + s];
      if (other < sdata[tid]) sdata[tid] = other;
    }
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

static void ensure_temp(void** d_temp, std::size_t& cap_bytes, std::size_t need_bytes) {
  if (need_bytes <= cap_bytes) return;
  if (*d_temp) { cudaFree(*d_temp); *d_temp = nullptr; }
  CUDA_CHECK(cudaMalloc(d_temp, need_bytes));
  cap_bytes = need_bytes;
}

static PartBResult run_partB_impl(const std::vector<i32>& input, bool use_shared_reduction,
                                  bool save_files, const std::string& outdir, int repeats,
                                  const std::string& tag) {
  ensure_dir(outdir);

  const u32 n = static_cast<u32>(input.size());
  PartBResult res;
  if (n < 2) return res;

  DeviceBuffer<i32> d_in(n);
  CUDA_CHECK(cudaMemcpy(d_in.data(), input.data(), n * sizeof(i32), cudaMemcpyHostToDevice));

  DeviceBuffer<i32> d_sorted0(n), d_sorted1(n);

  void* d_temp = nullptr;
  std::size_t temp_cap = 0;

  auto do_one = [&]() -> float {
    // copy input into sort buffer
    CUDA_CHECK(cudaMemcpy(d_sorted0.data(), d_in.data(), n * sizeof(i32), cudaMemcpyDeviceToDevice));

    GpuTimer timer;
    timer.start();

    // sort ascending using DoubleBuffer
    cub::DoubleBuffer<i32> buf(d_sorted0.data(), d_sorted1.data());
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, buf, n));
    ensure_temp(&d_temp, temp_cap, temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, buf, n));
    CUDA_SYNC_CHECK();

    const i32* sorted = buf.Current();

    u64 best_key_h = pack_key(0xFFFFFFFFu, 0u);

    if (!use_shared_reduction) {
      DeviceBuffer<u64> d_best(1);
      CUDA_CHECK(cudaMemcpy(d_best.data(), &best_key_h, sizeof(u64), cudaMemcpyHostToDevice));
      int threads = PARTB_BLOCK_THREADS;
      int blocks = 256;
      minpair_atomic_kernel<<<blocks, threads>>>(sorted, n, d_best.data());
      CUDA_SYNC_CHECK();
      CUDA_CHECK(cudaMemcpy(&best_key_h, d_best.data(), sizeof(u64), cudaMemcpyDeviceToHost));
    } else {
      int threads = PARTB_BLOCK_THREADS;
      int blocks = 256;
      DeviceBuffer<u64> d_block(blocks);
      minpair_blockreduce_kernel<<<blocks, threads>>>(sorted, n, d_block.data());
      CUDA_SYNC_CHECK();

      u32 cur_n = blocks;
      DeviceBuffer<u64> d_in_keys(cur_n);
      CUDA_CHECK(cudaMemcpy(d_in_keys.data(), d_block.data(), cur_n * sizeof(u64), cudaMemcpyDeviceToDevice));

      while (cur_n > 1) {
        u32 out_n = (cur_n + threads - 1) / threads;
        DeviceBuffer<u64> d_out_keys(out_n);
        reduce_keys_kernel<<<out_n, threads>>>(d_in_keys.data(), cur_n, d_out_keys.data());
        CUDA_SYNC_CHECK();
        d_in_keys.swap(d_out_keys);
        cur_n = out_n;
      }

      CUDA_CHECK(cudaMemcpy(&best_key_h, d_in_keys.data(), sizeof(u64), cudaMemcpyDeviceToHost));
    }

    u32 best_diff = 0, best_i = 0;
    unpack_key(best_key_h, best_diff, best_i);

    i32 a = 0, b = 0;
    CUDA_CHECK(cudaMemcpy(&a, sorted + best_i, sizeof(i32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b, sorted + best_i + 1, sizeof(i32), cudaMemcpyDeviceToHost));

    res.a = a;
    res.b = b;
    res.diff = best_diff;

    return timer.stop_ms();
  };

  (void)do_one(); // warmup

  float best = 1e30f;
  for (int r = 0; r < std::max(1, repeats); ++r) best = std::min(best, do_one());
  res.ms_total = best;

  if (d_temp) cudaFree(d_temp);

  if (save_files) {
    std::string log_path = (std::filesystem::path(outdir) / "log.txt").string();
    std::ostringstream oss;
    oss << "=== Part B (" << tag << ") ===\n"
        << "N=" << n << " pair=(" << res.a << ", " << res.b << ") diff=" << res.diff << "\n"
        << "best_ms=" << std::fixed << std::setprecision(3) << res.ms_total << "\n\n";
    append_text(log_path, oss.str());

    std::string out_path = (std::filesystem::path(outdir) / ("partB_" + tag + "_result.txt")).string();
    std::ostringstream out;
    out << "a=" << res.a << "\n"
        << "b=" << res.b << "\n"
        << "diff=" << res.diff << "\n";
    append_text(out_path, out.str());
  }

  return res;
}

PartBResult run_partB_distributed(const std::vector<i32>& input, bool save_files,
                                 const std::string& outdir, int repeats) {
  return run_partB_impl(input, false, save_files, outdir, repeats, "distributed");
}

PartBResult run_partB_shared(const std::vector<i32>& input, bool save_files,
                             const std::string& outdir, int repeats) {
  return run_partB_impl(input, true, save_files, outdir, repeats, "shared");
}

}
