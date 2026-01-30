#include "io.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hw3 {

static inline bool is_sep(char c) {
  return c == ',' || c == '\n' || c == '\r' || c == '\t' || c == ' ';
}

std::vector<i32> read_dataset(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open input file: " + path);

  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (s.empty()) return {};

  std::vector<i32> out;
  out.reserve(1 << 20);

  const char* p = s.c_str();
  const char* e = p + s.size();

  while (p < e) {
    while (p < e && is_sep(*p)) ++p;
    if (p >= e) break;

    bool neg = false;
    if (*p == '-') { neg = true; ++p; }

    long long val = 0;
    bool any = false;
    while (p < e && *p >= '0' && *p <= '9') {
      any = true;
      val = val * 10 + (*p - '0');
      ++p;
    }
    if (!any) { ++p; continue; }
    if (neg) val = -val;
    out.push_back(static_cast<i32>(val));

    while (p < e && is_sep(*p)) ++p;
  }
  return out;
}

void write_csv_line(const std::string& path, const std::vector<i32>& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write file: " + path);
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (i) f << ",";
    f << data[i];
  }
  f << "\n";
}

void append_text(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::app);
  if (!f) throw std::runtime_error("Cannot write file: " + path);
  f << text;
}

void ensure_dir(const std::string& dir) {
  std::filesystem::create_directories(std::filesystem::path(dir));
}

}
