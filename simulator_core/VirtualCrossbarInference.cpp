// Ideal (noise-free) inference logic for VirtualCrossbar.

#include <stdexcept>
#include <string>

#include "VirtualCrossbar.h"

std::vector<int32_t> VirtualCrossbar::forward(const std::vector<int8_t>& input) const {
    if (input.size() != rows_) {
        throw std::invalid_argument("VirtualCrossbar '" + name_ + "': forward() expects " +
                                    std::to_string(rows_) + " inputs, got " +
                                    std::to_string(input.size()));
    }

    std::vector<int32_t> output(cols_, 0);

    // Walk the array row by row (contiguous in memory). With x[r] in {+1, -1}, the
    // multiply reduces to adding or subtracting the whole word-line row.
    for (std::size_t r = 0; r < rows_; ++r) {
        const int8_t x = input[r];
        if (x != 1 && x != -1) {
            throw std::invalid_argument("VirtualCrossbar '" + name_ + "': input[" +
                                        std::to_string(r) + "] must be +1 or -1");
        }
        const int8_t* row = &cells_[r * cols_];
        if (x == 1) {
            for (std::size_t c = 0; c < cols_; ++c) output[c] += row[c];
        } else {
            for (std::size_t c = 0; c < cols_; ++c) output[c] -= row[c];
        }
    }
    return output;
}

std::vector<int8_t> signActivation(const std::vector<int32_t>& preActivations) {
    std::vector<int8_t> out;
    out.reserve(preActivations.size());
    for (int32_t y : preActivations) out.push_back(y >= 0 ? 1 : -1);
    return out;
}
