// Unit tests for the Noise Injector and the per-device MTJCrossbar interface.
//
// Run from the repository root (CTest does this automatically) so that the trained
// weights and the 100 reference test vectors are found.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "CliOptions.h"
#include "CsvUtils.h"
#include "MTJCrossbar.h"
#include "MTJParams.h"
#include "NoiseInjector.h"
#include "VirtualCrossbar.h"

namespace {

using BitLine = MTJCrossbar::BitLine;

int g_failures = 0;
int g_checksInTest = 0;
int g_failuresInTest = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checksInTest;                                                                 \
        if (!(cond)) {                                                                    \
            ++g_failuresInTest;                                                           \
            if (g_failuresInTest <= 5) {                                                  \
                std::cerr << "    CHECK failed at line " << __LINE__ << ": " #cond "\n"; \
            }                                                                             \
        }                                                                                 \
    } while (0)

template <typename F>
bool throwsInvalidArgument(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void runTest(const std::string& name, const std::function<void()>& body) {
    g_checksInTest = 0;
    g_failuresInTest = 0;
    try {
        body();
    } catch (const std::exception& e) {
        ++g_failuresInTest;
        std::cerr << "    unexpected exception: " << e.what() << '\n';
    }
    if (g_failuresInTest == 0) {
        std::cout << "[PASS] " << name << " (" << g_checksInTest << " checks)\n";
    } else {
        ++g_failures;
        std::cout << "[FAIL] " << name << " (" << g_failuresInTest << " of " << g_checksInTest
                  << " checks failed)\n";
    }
}

// ---------------------------------------------------------------------------------
// Shared fixtures: the trained network and the 100 reference images.

struct Fixture {
    VirtualCrossbar layer1{"Layer1", 784, 256};
    VirtualCrossbar layer2{"Layer2", 256, 10};
    std::vector<std::vector<int8_t>> inputs;
    MTJParams params;

    Fixture() {
        layer1.loadFromCsv("layer1_weights_bin.csv");
        layer2.loadFromCsv("layer2_weights_bin.csv");
        for (const auto& row : readIntCsv("test_inputs_pm1.csv")) {
            inputs.emplace_back(row.begin(), row.end());
        }
    }
};

const Fixture& fixture() {
    static const Fixture f;
    return f;
}

std::vector<double> allConductances(const MTJCrossbar& x) {
    std::vector<double> g;
    g.reserve(x.deviceCount());
    for (std::size_t r = 0; r < x.rows(); ++r) {
        for (std::size_t c = 0; c < x.cols(); ++c) {
            g.push_back(x.conductance(r, c, BitLine::Positive));
            g.push_back(x.conductance(r, c, BitLine::Negative));
        }
    }
    return g;
}

std::size_t integerClass(const VirtualCrossbar& l1, const VirtualCrossbar& l2,
                         const std::vector<int8_t>& x) {
    const std::vector<int32_t> y = l2.forward(signActivation(l1.forward(x)));
    return static_cast<std::size_t>(std::max_element(y.begin(), y.end()) - y.begin());
}

std::size_t analogClass(const MTJCrossbar& m1, const MTJCrossbar& m2,
                        const std::vector<int8_t>& x, double unit) {
    const std::vector<int8_t> hidden = senseSign(m1.forward(x).difference(), unit);
    return senseArgmax(m2.forward(hidden).difference(), unit);
}

NoiseInjector makeInjector(std::uint64_t seed, double sigma, double p, FlipMode mode) {
    NoiseInjector inj(seed);
    inj.add(std::make_unique<ResistanceVariation>(sigma));
    inj.add(std::make_unique<BitFlip>(p, mode));
    return inj;
}

// ---------------------------------------------------------------------------------

void testZeroNoiseMatchesIntegerModel() {
    const Fixture& f = fixture();
    const double unit = f.params.unitCurrent();
    for (FlipMode mode : {FlipMode::PerCell, FlipMode::WholeWeight}) {
        MTJCrossbar m1(f.layer1, f.params);
        MTJCrossbar m2(f.layer2, f.params);
        const MTJCrossbar clean1(f.layer1, f.params);
        const MTJCrossbar clean2(f.layer2, f.params);

        NoiseInjector inj = makeInjector(42, 0.0, 0.0, mode);
        inj.apply(m1);
        inj.apply(m2);

        CHECK(allConductances(m1) == allConductances(clean1));  // bit-identical
        CHECK(allConductances(m2) == allConductances(clean2));
        CHECK(f.inputs.size() == 100);
        for (const auto& x : f.inputs) {
            CHECK(analogClass(m1, m2, x, unit) == integerClass(f.layer1, f.layer2, x));
        }
    }
}

// p = 1 in whole-weight mode inverts every weight, so every column sum becomes -y.
void testFullWholeWeightFlipNegatesSums() {
    const Fixture& f = fixture();
    const double unit = f.params.unitCurrent();
    MTJCrossbar m1(f.layer1, f.params);
    MTJCrossbar m2(f.layer2, f.params);
    NoiseInjector inj = makeInjector(7, 0.0, 1.0, FlipMode::WholeWeight);
    inj.apply(m1);
    inj.apply(m2);

    for (const auto& x : f.inputs) {
        const std::vector<int32_t> y1 = f.layer1.forward(x);
        const std::vector<double> d1 = m1.forward(x).difference();
        for (std::size_t c = 0; c < y1.size(); ++c) {
            CHECK(std::fabs(d1[c] / unit + static_cast<double>(y1[c])) < 1e-9);
        }
        const std::vector<int8_t> h = signActivation(y1);
        const std::vector<int32_t> y2 = f.layer2.forward(h);
        const std::vector<double> d2 = m2.forward(h).difference();
        for (std::size_t c = 0; c < y2.size(); ++c) {
            CHECK(std::fabs(d2[c] / unit + static_cast<double>(y2[c])) < 1e-9);
        }
    }
}

// p = 1 per-cell flips both devices of every pair: identical to whole-weight p = 1.
void testFullPerCellFlipEqualsWholeWeight() {
    const Fixture& f = fixture();
    for (const VirtualCrossbar* layer : {&f.layer1, &f.layer2}) {
        MTJCrossbar perCell(*layer, f.params);
        MTJCrossbar whole(*layer, f.params);
        makeInjector(1, 0.0, 1.0, FlipMode::PerCell).apply(perCell);
        makeInjector(2, 0.0, 1.0, FlipMode::WholeWeight).apply(whole);
        CHECK(allConductances(perCell) == allConductances(whole));
        for (std::size_t r = 0; r < layer->rows(); ++r) {
            for (std::size_t c = 0; c < layer->cols(); ++c) {
                const bool plus = layer->at(r, c) == 1;
                CHECK(perCell.isParallel(r, c, BitLine::Positive) == !plus);
                CHECK(perCell.isParallel(r, c, BitLine::Negative) == plus);
            }
        }
    }
}

// Flipping one device of one pair removes exactly that weight's contribution
// x_r * w_rc from its column and leaves all other columns unchanged.
void testSingleCellFlipZeroesWeight() {
    const Fixture& f = fixture();
    const double unit = f.params.unitCurrent();
    const std::vector<int8_t> x = signActivation(f.layer1.forward(f.inputs[0]));

    // Pick one +1 and one -1 weight in layer 2, and flip each device in turn.
    std::vector<std::pair<std::size_t, std::size_t>> picks;
    for (int8_t want : {int8_t{1}, int8_t{-1}}) {
        bool found = false;
        for (std::size_t r = 0; r < f.layer2.rows() && !found; ++r) {
            for (std::size_t c = 0; c < f.layer2.cols() && !found; ++c) {
                if (f.layer2.at(r, c) == want) {
                    picks.emplace_back(r, c);
                    found = true;
                }
            }
        }
        CHECK(found);
    }

    const MTJCrossbar clean(f.layer2, f.params);
    const std::vector<double> before = clean.forward(x).difference();
    const std::vector<int32_t> y = f.layer2.forward(x);

    for (const auto& [r, c] : picks) {
        const double contribution = static_cast<double>(x[r] * f.layer2.at(r, c));
        for (BitLine line : {BitLine::Positive, BitLine::Negative}) {
            MTJCrossbar m(f.layer2, f.params);
            m.flipCell(r, c, line);
            // The pair is now P/P or AP/AP: equal conductances, zero differential.
            CHECK(m.isParallel(r, c, BitLine::Positive) == m.isParallel(r, c, BitLine::Negative));
            CHECK(m.conductance(r, c, BitLine::Positive) == m.conductance(r, c, BitLine::Negative));

            const std::vector<double> after = m.forward(x).difference();
            for (std::size_t k = 0; k < after.size(); ++k) {
                if (k == c) {
                    CHECK(std::fabs((before[k] - after[k]) / unit - contribution) < 1e-9);
                    CHECK(std::fabs(after[k] / unit - (static_cast<double>(y[k]) - contribution)) <
                          1e-9);
                } else {
                    CHECK(after[k] == before[k]);  // bit-identical
                }
            }

            // Flipping the same device back restores the original exactly.
            m.flipCell(r, c, line);
            CHECK(allConductances(m) == allConductances(clean));
        }
    }
}

void testSeedReproducibility() {
    const Fixture& f = fixture();
    for (FlipMode mode : {FlipMode::PerCell, FlipMode::WholeWeight}) {
        MTJCrossbar a1(f.layer1, f.params), a2(f.layer2, f.params);
        MTJCrossbar b1(f.layer1, f.params), b2(f.layer2, f.params);
        MTJCrossbar c1(f.layer1, f.params), c2(f.layer2, f.params);

        NoiseInjector a = makeInjector(1234, 0.1, 0.05, mode);
        NoiseInjector b = makeInjector(1234, 0.1, 0.05, mode);
        NoiseInjector c = makeInjector(1235, 0.1, 0.05, mode);
        a.apply(a1);
        a.apply(a2);
        b.apply(b1);
        b.apply(b2);
        c.apply(c1);
        c.apply(c2);

        CHECK(allConductances(a1) == allConductances(b1));
        CHECK(allConductances(a2) == allConductances(b2));
        CHECK(allConductances(a1) != allConductances(c1));
        CHECK(allConductances(a2) != allConductances(c2));

        // Identical noisy conductances give identical predictions.
        const double unit = f.params.unitCurrent();
        for (const auto& x : f.inputs) {
            CHECK(analogClass(a1, a2, x, unit) == analogClass(b1, b2, x, unit));
        }
    }

    // The raw random stream is reproducible as well.
    NoiseRng r1(99), r2(99);
    for (int i = 0; i < 1000; ++i) CHECK(r1.standardNormal() == r2.standardNormal());
}

void testInvalidParametersThrow() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    CHECK(throwsInvalidArgument([] { ResistanceVariation(-0.1); }));
    CHECK(throwsInvalidArgument([&] { ResistanceVariation{nan}; }));
    CHECK(throwsInvalidArgument([&] { ResistanceVariation{inf}; }));
    CHECK(!throwsInvalidArgument([] { ResistanceVariation(0.0); }));

    CHECK(throwsInvalidArgument([] { BitFlip(-0.01); }));
    CHECK(throwsInvalidArgument([] { BitFlip(1.01); }));
    CHECK(throwsInvalidArgument([&] { BitFlip{nan}; }));
    CHECK(throwsInvalidArgument([] { BitFlip(0.5, static_cast<FlipMode>(7)); }));
    CHECK(!throwsInvalidArgument([] { BitFlip(0.0); }));
    CHECK(!throwsInvalidArgument([] { BitFlip(1.0, FlipMode::WholeWeight); }));

    CHECK(throwsInvalidArgument([] { parseFlipMode("foo"); }));
    CHECK(throwsInvalidArgument([] { parseFlipMode(""); }));
    CHECK(parseFlipMode("cell") == FlipMode::PerCell);
    CHECK(parseFlipMode("weight") == FlipMode::WholeWeight);

    NoiseInjector inj(0);
    CHECK(throwsInvalidArgument([&] { inj.add(nullptr); }));

    MTJCrossbar m(fixture().layer2, fixture().params);
    CHECK(throwsInvalidArgument([&] { m.setConductance(0, 0, BitLine::Positive, 0.0); }));
    CHECK(throwsInvalidArgument([&] { m.setConductance(0, 0, BitLine::Positive, nan); }));

    using Args = std::vector<std::string>;
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--sigma", "-0.1"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--sigma", "abc"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--p", "1.5"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--p", "0.1x"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--flip-mode", "foo"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--seed", "-3"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--seed"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"--bogus", "1"}); }));
    CHECK(throwsInvalidArgument([] { parseCliOptions(Args{"a", "b", "c", "d"}); }));
}

void testCliDefaultsAndPositionals() {
    using Args = std::vector<std::string>;
    const CliOptions d = parseCliOptions(Args{});
    CHECK(d.layer1Path == "layer1_weights_bin.csv");
    CHECK(d.layer2Path == "layer2_weights_bin.csv");
    CHECK(d.testDir.empty());
    CHECK(!d.noiseEnabled());
    CHECK(d.flipMode == FlipMode::PerCell);

    const CliOptions o = parseCliOptions(
        Args{"l1.csv", "--sigma", "0.2", "l2.csv", "--p", "0.01", "dir", "--flip-mode", "weight",
             "--seed", "77"});
    CHECK(o.layer1Path == "l1.csv");
    CHECK(o.layer2Path == "l2.csv");
    CHECK(o.testDir == "dir/");
    CHECK(o.sigma == 0.2);
    CHECK(o.p == 0.01);
    CHECK(o.flipMode == FlipMode::WholeWeight);
    CHECK(o.seed == 77);
    CHECK(o.noiseEnabled());
}

// Statistical sanity on layer 1 (401,408 devices, fixed seed, so deterministic).
void testNoiseStatistics() {
    const Fixture& f = fixture();
    const MTJCrossbar clean(f.layer1, f.params);
    const double n = static_cast<double>(clean.deviceCount());

    // ln(G0 / G) = ln(R / R0) = sigma * z: mean 0, standard deviation sigma.
    const double sigma = 0.2;
    MTJCrossbar varied(f.layer1, f.params);
    NoiseInjector(5).apply(varied);  // empty injector: no change
    NoiseInjector rv(5);
    rv.add(std::make_unique<ResistanceVariation>(sigma));
    rv.apply(varied);
    const std::vector<double> g0 = allConductances(clean);
    const std::vector<double> g = allConductances(varied);
    double sum = 0.0, sumSq = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double lr = std::log(g0[i] / g[i]);
        sum += lr;
        sumSq += lr * lr;
    }
    const double mean = sum / n;
    const double sd = std::sqrt(sumSq / n - mean * mean);
    // Standard error of the mean is sigma / sqrt(n) ~ 3e-4; allow about 5 standard errors.
    CHECK(std::fabs(mean) < 1.5e-3);
    CHECK(std::fabs(sd - sigma) < 1.5e-3);

    // Measured flip rate in both modes.
    const double p = 0.05;
    for (FlipMode mode : {FlipMode::PerCell, FlipMode::WholeWeight}) {
        MTJCrossbar flipped(f.layer1, f.params);
        NoiseInjector bf(6);
        bf.add(std::make_unique<BitFlip>(p, mode));
        bf.apply(flipped);
        std::size_t flips = 0, pairsSplit = 0;
        for (std::size_t r = 0; r < flipped.rows(); ++r) {
            for (std::size_t c = 0; c < flipped.cols(); ++c) {
                const bool pos = flipped.isParallel(r, c, BitLine::Positive) !=
                                 clean.isParallel(r, c, BitLine::Positive);
                const bool neg = flipped.isParallel(r, c, BitLine::Negative) !=
                                 clean.isParallel(r, c, BitLine::Negative);
                flips += static_cast<std::size_t>(pos) + static_cast<std::size_t>(neg);
                if (pos != neg) ++pairsSplit;
            }
        }
        // Standard error of the rate ~ sqrt(p(1-p)/n) ~ 3.4e-4 (5e-4 per weight).
        const double rate = static_cast<double>(flips) / n;
        CHECK(std::fabs(rate - p) < 2.5e-3);
        if (mode == FlipMode::WholeWeight) {
            CHECK(pairsSplit == 0);  // devices of a pair always flip together
        } else {
            CHECK(pairsSplit > 0);
        }
    }
}

}  // namespace

int main() {
    try {
        fixture();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] cannot load reference data (run from the repository root): "
                  << e.what() << '\n';
        return 1;
    }

    runTest("sigma = 0 and p = 0 match the integer model on 100 images",
            testZeroNoiseMatchesIntegerModel);
    runTest("p = 1 whole-weight negates every column sum", testFullWholeWeightFlipNegatesSums);
    runTest("p = 1 per-cell equals p = 1 whole-weight", testFullPerCellFlipEqualsWholeWeight);
    runTest("one flipped cell removes exactly that weight's contribution",
            testSingleCellFlipZeroesWeight);
    runTest("same seed reproduces, different seed differs", testSeedReproducibility);
    runTest("invalid parameters throw", testInvalidParametersThrow);
    runTest("command-line defaults and positional arguments", testCliDefaultsAndPositionals);
    runTest("noise statistics match sigma and p", testNoiseStatistics);

    if (g_failures != 0) {
        std::cout << g_failures << " test(s) failed.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
