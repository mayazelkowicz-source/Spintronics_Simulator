// Noise Injector: seeded random source and the resistance-variation and bit-flip
// noise models.

#include "NoiseInjector.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

using BitLine = MTJCrossbar::BitLine;

}  // namespace

double NoiseRng::uniform01() {
    // Top 53 bits of one 64-bit output, scaled by 2^-53.
    return static_cast<double>(engine_() >> 11) * 0x1.0p-53;
}

double NoiseRng::standardNormal() {
    const double u1 = 1.0 - uniform01();  // (0, 1], so log(u1) is finite
    const double u2 = uniform01();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

// ---------------------------------------------------------------------------------

ResistanceVariation::ResistanceVariation(double sigma) : sigma_(sigma) {
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::invalid_argument("ResistanceVariation: sigma must be finite and >= 0");
    }
}

std::string ResistanceVariation::describe() const {
    std::ostringstream os;
    os << "resistance variation, lognormal sigma = " << sigma_;
    return os.str();
}

void ResistanceVariation::apply(MTJCrossbar& xbar, NoiseRng& rng) const {
    if (sigma_ == 0.0) return;  // exact identity, no random numbers consumed
    for (std::size_t r = 0; r < xbar.rows(); ++r) {
        for (std::size_t c = 0; c < xbar.cols(); ++c) {
            for (BitLine line : {BitLine::Positive, BitLine::Negative}) {
                // R' = R * exp(sigma * z)  <=>  G' = G * exp(-sigma * z)
                const double factor = std::exp(-sigma_ * rng.standardNormal());
                xbar.setConductance(r, c, line, xbar.conductance(r, c, line) * factor);
            }
        }
    }
}

// ---------------------------------------------------------------------------------

FlipMode parseFlipMode(const std::string& text) {
    if (text == "cell") return FlipMode::PerCell;
    if (text == "weight") return FlipMode::WholeWeight;
    throw std::invalid_argument("unknown flip mode '" + text + "' (expected cell or weight)");
}

std::string toString(FlipMode mode) {
    return mode == FlipMode::PerCell ? "cell" : "weight";
}

BitFlip::BitFlip(double p, FlipMode mode) : p_(p), mode_(mode) {
    if (!(p >= 0.0 && p <= 1.0)) {  // also rejects NaN
        throw std::invalid_argument("BitFlip: p must be between 0 and 1");
    }
    if (mode != FlipMode::PerCell && mode != FlipMode::WholeWeight) {
        throw std::invalid_argument("BitFlip: unknown flip mode");
    }
}

std::string BitFlip::describe() const {
    std::ostringstream os;
    os << "bit flips, p = " << p_ << ", mode = "
       << (mode_ == FlipMode::PerCell ? "per-cell" : "whole-weight");
    return os.str();
}

void BitFlip::apply(MTJCrossbar& xbar, NoiseRng& rng) const {
    if (p_ == 0.0) return;  // exact identity, no random numbers consumed
    for (std::size_t r = 0; r < xbar.rows(); ++r) {
        for (std::size_t c = 0; c < xbar.cols(); ++c) {
            if (mode_ == FlipMode::PerCell) {
                for (BitLine line : {BitLine::Positive, BitLine::Negative}) {
                    if (rng.uniform01() < p_) xbar.flipCell(r, c, line);
                }
            } else if (rng.uniform01() < p_) {
                xbar.flipCell(r, c, BitLine::Positive);
                xbar.flipCell(r, c, BitLine::Negative);
            }
        }
    }
}

// ---------------------------------------------------------------------------------

void NoiseInjector::add(std::unique_ptr<NoiseModel> model) {
    if (!model) throw std::invalid_argument("NoiseInjector: null noise model");
    models_.push_back(std::move(model));
}

void NoiseInjector::apply(MTJCrossbar& xbar) {
    for (const auto& model : models_) model->apply(xbar, rng_);
}
