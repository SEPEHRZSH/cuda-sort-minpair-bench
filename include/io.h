#pragma once
#include "config.h"
#include <string>
#include <vector>

namespace hw3 {

std::vector<i32> read_dataset(const std::string& path);
void write_csv_line(const std::string& path, const std::vector<i32>& data);
void append_text(const std::string& path, const std::string& text);
void ensure_dir(const std::string& dir);

}
