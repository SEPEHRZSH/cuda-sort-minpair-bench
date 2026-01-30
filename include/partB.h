#pragma once
#include "config.h"
#include <string>
#include <vector>

namespace hw3 {

struct PartBResult {
  i32 a = 0;
  i32 b = 0;
  u32 diff = 0;
  float ms_total = 0.0f;
};

PartBResult run_partB_distributed(const std::vector<i32>& input, bool save_files,
                                 const std::string& outdir, int repeats);

PartBResult run_partB_shared(const std::vector<i32>& input, bool save_files,
                             const std::string& outdir, int repeats);

}
