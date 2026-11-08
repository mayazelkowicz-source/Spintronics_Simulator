#ifndef NOISE_INJECTOR_H
#define NOISE_INJECTOR_H

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "MTJCrossbar.h"

// Seeded random source for all noise models.
//
// The engine is std::mt19937_64, whose output sequence is fixed by the C++ standard.
// The conversion to uniform and normal numbers is done here instead of with
// std::uniform_real_distribution / std::normal_distribution, whose outputs differ
// between standard libraries. The same seed therefore gives the same noise on every
// platform and compiler.
class NoiseRng {
public:
    explicit NoiseRng(std::uint64_t seed) : engine_(seed) {}

    // Uniform in [0, 1) with 53 random bits.
    double uniform01();

    // Standard normal N(0, 1), Box-Muller transform. Each call consumes two
    // uniform draws, so the sequence does not depend on any cached state.
    double standardNormal();

private:
    std::mt19937_64 engine_;
};

// Interface for a noise model. A model modifies the stored per-device conductances
// and states of an MTJCrossbar; MTJCrossbar::forward() then reads the noisy array.
// New models are added by deriving from this class and registering an instance
// with NoiseInjector::add(), without changes to the crossbar or the sense logic.
class NoiseModel {
public:
    virtual ~NoiseModel() = default;
    virtual std::string describe() const = 0;
    virtual void apply(MTJCrossbar& xbar, NoiseRng& rng) const = 0;
};

// Device-to-device resistance variation: every device's resistance is multiplied by
// an independent lognormal factor exp(sigma * z), z ~ N(0, 1). The factor has
// median 1; sigma is the standard deviation of ln R. sigma = 0 leaves the crossbar
// untouched. Throws std::invalid_argument if sigma is negative or not finite.
class ResistanceVariation : public NoiseModel {
public:
    explicit ResistanceVariation(double sigma);
    std::string describe() const override;
    void apply(MTJCrossbar& xbar, NoiseRng& rng) const override;
    double sigma() const { return sigma_; }

private:
    double sigma_;
};

// PerCell:     each MTJ device flips between P and AP independently with
//              probability p. One flipped device of a pair makes the weight
//              contribute 0; both flipped invert it. Models independent device errors.
// WholeWeight: each weight flips with probability p by flipping both devices of its
//              pair, i.e. the weight is inverted. Matches the bit-error injection of
//              Hirtzlin et al. (2019).
enum class FlipMode { PerCell, WholeWeight };

// "cell" -> PerCell, "weight" -> WholeWeight. Throws std::invalid_argument otherwise.
FlipMode parseFlipMode(const std::string& text);
std::string toString(FlipMode mode);

// Bit flips with probability p. p = 0 leaves the crossbar untouched.
// Throws std::invalid_argument unless 0 <= p <= 1.
class BitFlip : public NoiseModel {
public:
    explicit BitFlip(double p, FlipMode mode = FlipMode::PerCell);
    std::string describe() const override;
    void apply(MTJCrossbar& xbar, NoiseRng& rng) const override;
    double probability() const { return p_; }
    FlipMode mode() const { return mode_; }

private:
    double p_;
    FlipMode mode_;
};

// Applies a list of noise models, in the order they were added, with one seeded
// random source. Crossbars are processed in the order apply() is called; within a
// crossbar, models visit cells row-major, positive device before negative device.
class NoiseInjector {
public:
    explicit NoiseInjector(std::uint64_t seed) : seed_(seed), rng_(seed) {}

    void add(std::unique_ptr<NoiseModel> model);
    void apply(MTJCrossbar& xbar);

    bool empty() const { return models_.empty(); }
    std::uint64_t seed() const { return seed_; }
    const std::vector<std::unique_ptr<NoiseModel>>& models() const { return models_; }

private:
    std::uint64_t seed_;
    NoiseRng rng_;
    std::vector<std::unique_ptr<NoiseModel>> models_;
};

#endif  // NOISE_INJECTOR_H
