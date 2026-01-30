#pragma once
#include "config.h"
#include <string>
#include <vector>

namespace hw3 {

bool verify_partA(const std::vector<i32>& out, u32 even_count);
std::string preview_vec(const std::vector<i32>& v, std::size_t k);

}
