// Analog MTJ crossbar model: weight mapping to P/AP differential pairs, bit-line
// current summation and sense-amplifier decisions.

#include "MTJCrossbar.h"

#include <cmath>
#include <stdexcept>
#include <string>

std::vector<double> BitLineCurrents::difference() const {
    std::vector<double> diff(pos.size());
    for (std::size_t c = 0; c < pos.size(); ++c) diff[c] = pos[c] - neg[c];
    return diff;
}

MTJCrossbar::MTJCrossbar(const VirtualCrossbar& weights, const MTJParams& params)
    : name_(weights.name()),
      rows_(weights.rows()),
      cols_(weights.cols()),
      params_(params),
      gPos_(rows_ * cols_),
      gNeg_(rows_ * cols_),
      parallelPos_(rows_ * cols_),
      parallelNeg_(rows_ * cols_) {
    params_.validate();
    const double gP = params_.gP();
    const double gAP = params_.gAP();

    // Weight Mapper: +1 -> (P, AP), -1 -> (AP, P) on the (positive, negative) bit lines.
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            const bool plus = weights.at(r, c) == 1;
            const std::size_t i = r * cols_ + c;
            gPos_[i] = plus ? gP : gAP;
            gNeg_[i] = plus ? gAP : gP;
            parallelPos_[i] = plus ? 1 : 0;
            parallelNeg_[i] = plus ? 0 : 1;
        }
    }
}

std::size_t MTJCrossbar::index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("MTJCrossbar '" + name_ + "': cell (" + std::to_string(row) +
                                ", " + std::to_string(col) + ") out of range");
    }
    return row * cols_ + col;
}

double MTJCrossbar::conductance(std::size_t row, std::size_t col, BitLine line) const {
    const std::size_t i = index(row, col);
    return line == BitLine::Positive ? gPos_[i] : gNeg_[i];
}

void MTJCrossbar::setConductance(std::size_t row, std::size_t col, BitLine line, double g) {
    const std::size_t i = index(row, col);
    if (!std::isfinite(g) || !(g > 0.0)) {
        throw std::invalid_argument("MTJCrossbar '" + name_ +
                                    "': conductance must be finite and > 0");
    }
    (line == BitLine::Positive ? gPos_[i] : gNeg_[i]) = g;
}

bool MTJCrossbar::isParallel(std::size_t row, std::size_t col, BitLine line) const {
    const std::size_t i = index(row, col);
    return (line == BitLine::Positive ? parallelPos_[i] : parallelNeg_[i]) != 0;
}

void MTJCrossbar::flipCell(std::size_t row, std::size_t col, BitLine line) {
    const std::size_t i = index(row, col);
    double& g = line == BitLine::Positive ? gPos_[i] : gNeg_[i];
    uint8_t& parallel = line == BitLine::Positive ? parallelPos_[i] : parallelNeg_[i];
    const double ratio = 1.0 + params_.tmr;  // R_AP / R_P = G_P / G_AP
    g = parallel ? g / ratio : g * ratio;
    parallel = parallel ? 0 : 1;
}

BitLineCurrents MTJCrossbar::forward(const std::vector<int8_t>& input) const {
    if (input.size() != rows_) {
        throw std::invalid_argument("MTJCrossbar '" + name_ + "': forward() expects " +
                                    std::to_string(rows_) + " inputs, got " +
                                    std::to_string(input.size()));
    }

    BitLineCurrents out{std::vector<double>(cols_, 0.0), std::vector<double>(cols_, 0.0)};

    // Each word line r is driven at v = +/-V_read; every cell on it contributes
    // I = v * G to its bit line (Ohm's law, ideal wires).
    for (std::size_t r = 0; r < rows_; ++r) {
        const int8_t x = input[r];
        if (x != 1 && x != -1) {
            throw std::invalid_argument("MTJCrossbar '" + name_ + "': input[" +
                                        std::to_string(r) + "] must be +1 or -1");
        }
        const double v = x * params_.vRead;
        const double* gp = &gPos_[r * cols_];
        const double* gn = &gNeg_[r * cols_];
        for (std::size_t c = 0; c < cols_; ++c) {
            out.pos[c] += v * gp[c];
            out.neg[c] += v * gn[c];
        }
    }
    return out;
}

std::vector<int8_t> senseSign(const std::vector<double>& diff, double unitCurrent) {
    const double threshold = -0.5 * unitCurrent;
    std::vector<int8_t> out;
    out.reserve(diff.size());
    for (double d : diff) out.push_back(d >= threshold ? 1 : -1);
    return out;
}

std::size_t senseArgmax(const std::vector<double>& diff, double unitCurrent) {
    if (diff.empty()) throw std::invalid_argument("senseArgmax: empty input");
    const double margin = 0.5 * unitCurrent;
    std::size_t best = 0;
    for (std::size_t c = 1; c < diff.size(); ++c) {
        if (diff[c] > diff[best] + margin) best = c;
    }
    return best;
}
