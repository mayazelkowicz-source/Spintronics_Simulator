#ifndef VIRTUAL_CROSSBAR_H
#define VIRTUAL_CROSSBAR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A 2D array of memory cells, each holding a binary weight in {+1, -1}.
//
// Layout follows the crossbar convention used by the training export
// (PythonProject/main.py): rows = input lines (word lines), columns = output
// lines (bit lines). Cell (r, c) holds W[c][r] of the corresponding nn.Linear.
class VirtualCrossbar {
public:
    VirtualCrossbar(std::string name, std::size_t rows, std::size_t cols);

    // Loads a CSV weight matrix into the crossbar. Accepted cell encodings:
    //   1 -> +1,  0 -> -1   (the *_weights_bin.csv files)
    //   1 -> +1, -1 -> -1   (the *_weights.csv files)
    // Throws std::runtime_error on I/O errors, bad tokens or a shape mismatch.
    void loadFromCsv(const std::string& path);

    int8_t at(std::size_t row, std::size_t col) const;
    void set(std::size_t row, std::size_t col, int8_t value);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::string& name() const { return name_; }

    // Ideal, noise-free forward pass (implemented in VirtualCrossbarInference.cpp).
    // Computes the column sums y[c] = sum_r input[r] * W[r][c], i.e. y = x @ W,
    // which is what each bit line of the crossbar accumulates.
    // input must have rows() entries, each +1 or -1; the result has cols() entries.
    std::vector<int32_t> forward(const std::vector<int8_t>& input) const;

    // Prints the top-left (sampleRows x sampleCols) corner of the array.
    void printSample(std::size_t sampleRows, std::size_t sampleCols) const;

private:
    std::size_t index(std::size_t row, std::size_t col) const;
    static int8_t parseCell(const std::string& token);

    std::string name_;
    std::size_t rows_;
    std::size_t cols_;
    std::vector<int8_t> cells_;  // row-major, contiguous
};

// Sign comparator applied between layers: +1 if y >= 0 else -1 (sign(0) -> +1),
// matching the training script's convention.
std::vector<int8_t> signActivation(const std::vector<int32_t>& preActivations);

#endif  // VIRTUAL_CROSSBAR_H
