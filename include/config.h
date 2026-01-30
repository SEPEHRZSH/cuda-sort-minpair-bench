#pragma once
#include <cstddef>
#include <cstdint>

namespace hw3 {
  using i32 = int32_t;
  using u32 = uint32_t;
  using u64 = uint64_t;

  constexpr int DEFAULT_REPEATS = 5;

  constexpr int PARTA_BLOCK_THREADS = 256;
  constexpr int PARTA_ITEMS_PER_THREAD = 4;

  constexpr int PARTB_BLOCK_THREADS = 256;
}
