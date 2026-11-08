// Stage 3: ideal integer inference plus the noise-free analog MTJ model.
//
// Loads both layers, runs the 784 -> 256 -> 10 BNN on the golden test vectors and
// checks the logits bit-exactly against the reference produced by the training
// script (numpy_golden_inference in main.py). The same network is then run on the
// analog MTJ crossbars (P/AP differential pairs, bit-line currents), which must
// predict the same class as the integer model for every sample.
//
// Usage: bnn_sim [layer1.csv] [layer2.csv] [test_dir]
// Defaults assume the binary is run from the project root.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "CsvUtils.h"
#include "MTJCrossbar.h"
#include "MTJParams.h"
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

// Largest |diff / unitCurrent - y| over all columns: how far the analog differential
// currents are from the exact integer column sums (floating-point error only).
double latticeDeviation(const std::vector<double>& diff, const std::vector<int32_t>& y,
                        double unitCurrent) {
    double worst = 0.0;
    for (std::size_t c = 0; c < diff.size(); ++c) {
        worst = std::max(worst, std::fabs(diff[c] / unitCurrent - static_cast<double>(y[c])));
    }
    return worst;
}

// The half-unit sense thresholds tolerate any error below 0.5 units; fail well before.
constexpr double kMaxLatticeDeviation = 0.25;

}  // namespace

int main(int argc, char* argv[]) {
    const std::string layer1Path = argc > 1 ? argv[1] : "layer1_weights_bin.csv";
    const std::string layer2Path = argc > 2 ? argv[2] : "layer2_weights_bin.csv";
    const std::string testDir = argc > 3 ? std::string(argv[3]) + "/" : "";

    try {
        VirtualCrossbar layer1("Layer1", kInputSize, kHiddenSize);
        VirtualCrossbar layer2("Layer2", kHiddenSize, kOutputSize);

        loadAndReport(layer1, layer1Path);
        loadAndReport(layer2, layer2Path);
        std::cout << "All crossbars loaded successfully.\n\n";

        const MTJParams mtjParams;  // PLACEHOLDER device values, see MTJParams.h
        const MTJCrossbar mtj1(layer1, mtjParams);
        const MTJCrossbar mtj2(layer2, mtjParams);
        const double unitCurrent = mtjParams.unitCurrent();
        std::cout << "MTJ device model (PLACEHOLDER values, not lab data):\n"
                  << "  V_read = " << mtjParams.vRead << " V, R_P = " << mtjParams.rP
                  << " ohm, TMR = " << 100.0 * mtjParams.tmr << "%, R_AP = " << mtjParams.rAP()
                  << " ohm\n"
                  << "  Unit current (one P/AP pair at V_read) = " << unitCurrent << " A\n"
                  << "  Physical MTJ devices (two per weight): " << mtj1.name() << " "
                  << mtj1.deviceCount() << ", " << mtj2.name() << " " << mtj2.deviceCount()
                  << ", total " << mtj1.deviceCount() + mtj2.deviceCount() << "\n\n";

        const auto inputs = readIntCsv(testDir + kTestInputsPath);
        const auto expected = readIntCsv(testDir + kExpectedLogitsPath);
        const auto labels = readIntCsv(testDir + kTestLabelsPath);
        if (expected.size() != inputs.size() || labels.size() != inputs.size()) {
            throw std::runtime_error("test files have mismatched sample counts");
        }

        std::size_t logitMatches = 0;
        std::size_t correct = 0;
        std::size_t analogClassMatches = 0;
        std::size_t analogCorrect = 0;
        double maxDeviation = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const std::vector<int8_t> x(inputs[i].begin(), inputs[i].end());
            const std::vector<int32_t> logits = infer(layer1, layer2, x);
            const std::size_t intClass = argmax(logits);

            if (logits == expected[i]) ++logitMatches;
            if (static_cast<int32_t>(intClass) == labels[i].at(0)) ++correct;

            // Analog path: bit-line currents -> sense amplifier -> bit-line currents.
            const std::vector<double> diff1 = mtj1.forward(x).difference();
            const std::vector<int8_t> hidden = senseSign(diff1, unitCurrent);
            const std::vector<double> diff2 = mtj2.forward(hidden).difference();
            const std::size_t analogClass = senseArgmax(diff2, unitCurrent);

            maxDeviation = std::max(maxDeviation,
                                    latticeDeviation(diff1, layer1.forward(x), unitCurrent));
            maxDeviation = std::max(maxDeviation,
                                    latticeDeviation(diff2, layer2.forward(hidden), unitCurrent));
            if (analogClass == intClass) ++analogClassMatches;
            if (static_cast<int32_t>(analogClass) == labels[i].at(0)) ++analogCorrect;
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        const std::size_t n = inputs.size();
        std::cout << "Ideal inference on " << n << " test samples:\n"
                  << "  Logits matching golden reference: " << logitMatches << " / " << n << '\n'
                  << "  Classification accuracy:          " << correct << " / " << n << " ("
                  << 100.0 * static_cast<double>(correct) / static_cast<double>(n) << "%)\n";

        if (logitMatches != n) {
            std::cerr << "[FAIL] Simulator logits diverge from the golden reference.\n";
            return 1;
        }
        std::cout << "[PASS] Simulator is bit-exact with the golden reference.\n\n";

        std::cout << "Analog MTJ inference (noise off) on " << n << " test samples:\n"
                  << "  Classes matching integer model:   " << analogClassMatches << " / " << n
                  << '\n'
                  << "  Classification accuracy:          " << analogCorrect << " / " << n << " ("
                  << 100.0 * static_cast<double>(analogCorrect) / static_cast<double>(n)
                  << "%)\n"
                  << "  Max deviation from integer sums:  " << maxDeviation
                  << " units (limit " << kMaxLatticeDeviation << ")\n"
                  << "  Run time (integer + analog):      " << seconds << " s\n";

        if (analogClassMatches != n || !(maxDeviation < kMaxLatticeDeviation)) {
            std::cerr << "[FAIL] Analog MTJ model diverges from the integer model.\n";
            return 1;
        }
        std::cout << "[PASS] Analog MTJ model predicts the same class as the integer model.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
