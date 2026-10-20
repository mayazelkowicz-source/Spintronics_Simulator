// Stage 2: ideal (noise-free) inference on the virtual crossbars.
//
// Loads both layers, runs the 784 -> 256 -> 10 BNN on the golden test vectors and
// checks the logits bit-exactly against the reference produced by the training
// script (numpy_golden_inference in main.py).
//
// Usage: bnn_sim [layer1.csv] [layer2.csv]
// Defaults assume the binary is run from the project root.

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "CsvUtils.h"
#include "VirtualCrossbar.h"

namespace {

// Network topology from main.py: 784 -> 256 -> 10.
constexpr std::size_t kInputSize = 784;
constexpr std::size_t kHiddenSize = 256;
constexpr std::size_t kOutputSize = 10;

const std::string kTestInputsPath = "test_inputs_pm1.csv";
const std::string kExpectedLogitsPath = "test_expected_logits.csv";
const std::string kTestLabelsPath = "test_labels.csv";

void loadAndReport(VirtualCrossbar& xbar, const std::string& path) {
    xbar.loadFromCsv(path);
    std::cout << "[OK] " << xbar.name() << ": loaded " << xbar.rows() << " x " << xbar.cols()
              << " weights from " << path << '\n';
    xbar.printSample(2, 2);
}

// Full network: crossbar 1 -> sign comparator -> crossbar 2 (raw logits).
std::vector<int32_t> infer(const VirtualCrossbar& layer1, const VirtualCrossbar& layer2,
                           const std::vector<int8_t>& input) {
    return layer2.forward(signActivation(layer1.forward(input)));
}

std::size_t argmax(const std::vector<int32_t>& v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
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
        std::cout << "All crossbars loaded successfully.\n\n";

        const auto inputs = readIntCsv(kTestInputsPath);
        const auto expected = readIntCsv(kExpectedLogitsPath);
        const auto labels = readIntCsv(kTestLabelsPath);
        if (expected.size() != inputs.size() || labels.size() != inputs.size()) {
            throw std::runtime_error("test files have mismatched sample counts");
        }

        std::size_t logitMatches = 0;
        std::size_t correct = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const std::vector<int8_t> x(inputs[i].begin(), inputs[i].end());
            const std::vector<int32_t> logits = infer(layer1, layer2, x);

            if (logits == expected[i]) ++logitMatches;
            if (static_cast<int32_t>(argmax(logits)) == labels[i].at(0)) ++correct;
        }

        const std::size_t n = inputs.size();
        std::cout << "Ideal inference on " << n << " test samples:\n"
                  << "  Logits matching golden reference: " << logitMatches << " / " << n << '\n'
                  << "  Classification accuracy:          " << correct << " / " << n << " ("
                  << 100.0 * static_cast<double>(correct) / static_cast<double>(n) << "%)\n";

        if (logitMatches != n) {
            std::cerr << "[FAIL] Simulator logits diverge from the golden reference.\n";
            return 1;
        }
        std::cout << "[PASS] Simulator is bit-exact with the golden reference.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
