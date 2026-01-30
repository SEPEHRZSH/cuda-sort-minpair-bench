#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace hw3 {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    (void)file;
  (void)line;
if (err != cudaSuccess) {
    std::string msg = std::string("CUDA error: ") + cudaGetErrorString(err) +
                      "\n  expr: " + expr +
                      "\n  at: " + file + ":" + std::to_string(line);
    throw std::runtime_error(msg);
  }
}

#define CUDA_CHECK(x) ::hw3::cuda_check((x), #x, __FILE__, __LINE__)

inline void cuda_sync_check(const char* file, int line) {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

#define CUDA_SYNC_CHECK() ::hw3::cuda_sync_check(__FILE__, __LINE__)

}