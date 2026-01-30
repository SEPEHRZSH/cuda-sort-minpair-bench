#pragma once
#include "config.h"
#include <string>
#include <vector>

namespace hw3 {

struct PartAResult {
  std::vector<i32> output;
  u32 even_count = 0;
  u32 odd_count = 0;
  float ms_total = 0.0f;
};

PartAResult run_partA_distributed(const std::vector<i32>& input, bool save_files,
                                 const std::string& outdir, int repeats);

PartAResult run_partA_shared(const std::vector<i32>& input, bool save_files,
                             const std::string& outdir, int repeats);

}