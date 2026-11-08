#ifndef CLI_OPTIONS_H
#define CLI_OPTIONS_H

#include <cstdint>
#include <string>
#include <vector>

#include "NoiseInjector.h"

// Command-line options of bnn_sim.
//
//   bnn_sim [layer1.csv] [layer2.csv] [test_dir]
//           [--sigma S] [--p P] [--flip-mode cell|weight] [--seed N] [--help]
//
// Positional arguments keep their original meaning and order; options may appear
// anywhere. With the defaults (sigma = 0, p = 0) noise is off.
struct CliOptions {
    std::string layer1Path = "layer1_weights_bin.csv";
    std::string layer2Path = "layer2_weights_bin.csv";
    std::string testDir;  // empty = repository root; otherwise ends with '/'

    double sigma = 0.0;
    double p = 0.0;
    FlipMode flipMode = FlipMode::PerCell;
    std::uint64_t seed = 1;
    bool help = false;

    bool noiseEnabled() const { return sigma > 0.0 || p > 0.0; }
};

// Parses the arguments after the program name. Throws std::invalid_argument on an
// unknown option, a missing or malformed value, an invalid sigma, p or flip mode,
// or more than three positional arguments.
CliOptions parseCliOptions(const std::vector<std::string>& args);

std::string cliUsage(const std::string& program);

#endif  // CLI_OPTIONS_H
