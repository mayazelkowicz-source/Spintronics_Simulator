#ifndef MTJ_CROSSBAR_H
#define MTJ_CROSSBAR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "MTJParams.h"
#include "VirtualCrossbar.h"

// Bit-line currents of one crossbar read: for every output column, the current on
// the positive bit line and on the negative bit line [A].
struct BitLineCurrents {
    std::vector<double> pos;
    std::vector<double> neg;

    // Differential current I_pos - I_neg per column, as seen by the sense amplifier.
    std::vector<double> difference() const;
};

// Analog model of a crossbar built from MTJ cells (Weight Mapper + MTJ Crossbar).
//
// Each binary weight is stored in a P/AP differential pair: one cell on a positive
// bit line and one on a negative bit line.
//   weight +1 -> positive cell Parallel (G_P),      negative cell Anti-Parallel (G_AP)
//   weight -1 -> positive cell Anti-Parallel (G_AP), negative cell Parallel (G_P)
// Inputs drive the word lines with +V_read (+1) or -V_read (-1). Each bit line sums
// the cell currents v_r * G, so I_pos - I_neg = V_read * (G_P - G_AP) * y, where y is
// the integer column sum of VirtualCrossbar::forward(). The input-dependent common
// current cancels in the difference.
//
// Conductances are stored per cell so that noise models can later modify
// individual devices.
class MTJCrossbar {
public:
    MTJCrossbar(const VirtualCrossbar& weights, const MTJParams& params);

    // Analog, noise-free read. input must have rows() entries, each +1 or -1.
    BitLineCurrents forward(const std::vector<int8_t>& input) const;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::string& name() const { return name_; }
    const MTJParams& params() const { return params_; }

    // Number of physical MTJ devices (two per weight).
    std::size_t deviceCount() const { return 2 * rows_ * cols_; }

private:
    std::string name_;
    std::size_t rows_;
    std::size_t cols_;
    MTJParams params_;
    std::vector<double> gPos_;  // row-major conductances on the positive bit lines [S]
    std::vector<double> gNeg_;  // row-major conductances on the negative bit lines [S]
};

// Sense amplifier between layers. Returns +1 if diff >= -0.5 * unitCurrent, else -1.
// Without noise, diff / unitCurrent is an integer column sum y, so the half-unit
// threshold reproduces sign(y) with sign(0) = +1 while tolerating floating-point
// rounding error smaller than half a unit.
std::vector<int8_t> senseSign(const std::vector<double>& diff, double unitCurrent);

// Output-layer decision. A column replaces the current best only if it is larger by
// more than 0.5 * unitCurrent, so exact ties keep the first maximum, matching the
// integer argmax (std::max_element).
std::size_t senseArgmax(const std::vector<double>& diff, double unitCurrent);

#endif  // MTJ_CROSSBAR_H
