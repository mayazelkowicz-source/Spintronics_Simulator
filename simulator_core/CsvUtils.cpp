#include "CsvUtils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::vector<int32_t>> readIntCsv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    std::vector<std::vector<int32_t>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF files
        if (line.empty()) continue;

        std::vector<int32_t> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                std::size_t used = 0;
                row.push_back(static_cast<int32_t>(std::stol(token, &used)));
                if (token.find_first_not_of(" \t", used) != std::string::npos) {
                    throw std::invalid_argument("trailing characters");
                }
            } catch (const std::exception&) {
                throw std::runtime_error(path + ": row " + std::to_string(rows.size()) +
                                         ": invalid integer '" + token + "'");
            }
        }
        if (!rows.empty() && row.size() != rows.front().size()) {
            throw std::runtime_error(path + ": row " + std::to_string(rows.size()) + " has " +
                                     std::to_string(row.size()) + " columns, expected " +
                                     std::to_string(rows.front().size()));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}
