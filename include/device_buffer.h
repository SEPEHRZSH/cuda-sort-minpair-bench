#pragma once
#include "cuda_check.h"
#include <cstddef>
#include <utility>

namespace hw3 {

template <class T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { allocate(n); }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) swap(other);
    return *this;
  }

  void allocate(std::size_t n) {
    reset();
    if (n == 0) return;
    n_ = n;
    CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
  }

  void reset() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      n_ = 0;
    }
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return n_; }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(n_, other.n_);
  }

private:
  T* ptr_ = nullptr;
  std::size_t n_ = 0;
};

}