#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

// Reads a comma-separated integer matrix, one row per non-empty line.
// Throws std::runtime_error on I/O errors, bad tokens or ragged rows.
std::vector<std::vector<int32_t>> readIntCsv(const std::string& path);

#endif  // CSV_UTILS_H
