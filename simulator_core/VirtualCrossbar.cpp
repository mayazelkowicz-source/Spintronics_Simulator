#include "VirtualCrossbar.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

VirtualCrossbar::VirtualCrossbar(std::string name, std::size_t rows, std::size_t cols)
    : name_(std::move(name)), rows_(rows), cols_(cols), cells_(rows * cols, -1) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("VirtualCrossbar '" + name_ + "': dimensions must be non-zero");
    }
}

void VirtualCrossbar::loadFromCsv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("VirtualCrossbar '" + name_ + "': cannot open " + path);
    }

    std::vector<int8_t> loaded;
    loaded.reserve(rows_ * cols_);

    std::string line;
    std::size_t row = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF files
        if (line.empty()) continue;

        if (row >= rows_) {
            throw std::runtime_error(path + ": more than " + std::to_string(rows_) + " rows");
        }

        std::stringstream ss(line);
        std::string token;
        std::size_t col = 0;
        while (std::getline(ss, token, ',')) {
            if (col >= cols_) {
                throw std::runtime_error(path + ": row " + std::to_string(row) +
                                         " has more than " + std::to_string(cols_) + " columns");
            }
            try {
                loaded.push_back(parseCell(token));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(path + ": row " + std::to_string(row) + ", col " +
                                         std::to_string(col) + ": " + e.what());
            }
            ++col;
        }
        if (col != cols_) {
            throw std::runtime_error(path + ": row " + std::to_string(row) + " has " +
                                     std::to_string(col) + " columns, expected " +
                                     std::to_string(cols_));
        }
        ++row;
    }

    if (row != rows_) {
        throw std::runtime_error(path + ": found " + std::to_string(row) + " rows, expected " +
                                 std::to_string(rows_));
    }

    // Commit only after the whole file validated, so a failed load leaves the array intact.
    cells_ = std::move(loaded);
}

int8_t VirtualCrossbar::at(std::size_t row, std::size_t col) const {
    return cells_[index(row, col)];
}

void VirtualCrossbar::set(std::size_t row, std::size_t col, int8_t value) {
    if (value != 1 && value != -1) {
        throw std::invalid_argument("VirtualCrossbar: cell value must be +1 or -1");
    }
    cells_[index(row, col)] = value;
}

void VirtualCrossbar::printSample(std::size_t sampleRows, std::size_t sampleCols) const {
    const std::size_t r = std::min(sampleRows, rows_);
    const std::size_t c = std::min(sampleCols, cols_);
    std::cout << "  " << name_ << " sample [0.." << r - 1 << "][0.." << c - 1 << "]:\n";
    for (std::size_t i = 0; i < r; ++i) {
        std::cout << "   ";
        for (std::size_t j = 0; j < c; ++j) {
            std::cout << std::setw(3) << static_cast<int>(at(i, j));
        }
        std::cout << '\n';
    }
}

std::size_t VirtualCrossbar::index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("VirtualCrossbar '" + name_ + "': index (" + std::to_string(row) +
                                ", " + std::to_string(col) + ") out of range");
    }
    return row * cols_ + col;
}

int8_t VirtualCrossbar::parseCell(const std::string& token) {
    // Trim surrounding whitespace.
    const auto first = token.find_first_not_of(" \t");
    const auto last = token.find_last_not_of(" \t");
    const std::string t = (first == std::string::npos) ? "" : token.substr(first, last - first + 1);

    if (t == "1" || t == "+1") return 1;
    if (t == "0" || t == "-1") return -1;
    throw std::runtime_error("invalid weight token '" + t + "'");
}
