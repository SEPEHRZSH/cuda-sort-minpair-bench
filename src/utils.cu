#include "utils.h"
#include <sstream>

namespace hw3 {

bool verify_partA(const std::vector<i32>& out, u32 even_count) {
  if (even_count > out.size()) return false;

  for (u32 i = 0; i < even_count; ++i) {
    if ((out[i] & 1) != 0) return false;
    if (i && out[i] < out[i - 1]) return false;
  }
  for (u32 i = even_count; i < out.size(); ++i) {
    if ((out[i] & 1) == 0) return false;
    if (i > even_count && out[i] > out[i - 1]) return false;
  }
  return true;
}

std::string preview_vec(const std::vector<i32>& v, std::size_t k) {
  std::ostringstream oss;
  oss << "[";
  std::size_t n = v.size();
  std::size_t m = (n < k) ? n : k;
  for (std::size_t i = 0; i < m; ++i) {
    if (i) oss << ", ";
    oss << v[i];
  }
  if (n > m) oss << ", ... (" << n << " items)";
  oss << "]";
  return oss.str();
}

}
