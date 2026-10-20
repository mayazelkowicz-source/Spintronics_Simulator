// Stage 1 sanity check: load the trained binary weights into virtual crossbars.
//
// Usage: bnn_sim [layer1.csv] [layer2.csv]
// Defaults assume the binary is run from the project root.

#include <exception>
#include <iostream>
#include <string>

#include "VirtualCrossbar.h"

namespace {

// Network topology from PythonProject/main.py: 784 -> 256 -> 10.
constexpr std::size_t kInputSize = 784;
constexpr std::size_t kHiddenSize = 256;
constexpr std::size_t kOutputSize = 10;

void loadAndReport(VirtualCrossbar& xbar, const std::string& path) {
    xbar.loadFromCsv(path);
    std::cout << "[OK] " << xbar.name() << ": loaded " << xbar.rows() << " x " << xbar.cols()
              << " weights from " << path << '\n';
    xbar.printSample(2, 2);
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string layer1Path = argc > 1 ? argv[1] : "layer1_weights_bin.csv";
    const std::string layer2Path = argc > 2 ? argv[2] : "layer2_weights_bin.csv";

    try {
        VirtualCrossbar layer1("Layer1", kInputSize, kHiddenSize);
        VirtualCrossbar layer2("Layer2", kHiddenSize, kOutputSize);

        loadAndReport(layer1, layer1Path);
        loadAndReport(layer2, layer2Path);

        std::cout << "All crossbars loaded successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
