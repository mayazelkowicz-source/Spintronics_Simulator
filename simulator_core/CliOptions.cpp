#include "CliOptions.h"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

double parseDouble(const std::string& option, const std::string& text) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(option + ": '" + text + "' is not a finite number");
    }
    return value;
}

std::uint64_t parseSeed(const std::string& text) {
    // Digits only: std::stoull would silently accept a leading '-' and wrap around.
    bool digits = !text.empty();
    for (char ch : text) digits = digits && std::isdigit(static_cast<unsigned char>(ch));
    std::size_t used = 0;
    unsigned long long value = 0;
    if (digits) {
        try {
            value = std::stoull(text, &used);
        } catch (const std::exception&) {
            used = 0;
        }
    }
    if (!digits || used != text.size()) {
        throw std::invalid_argument("--seed: '" + text + "' is not a non-negative integer");
    }
    return static_cast<std::uint64_t>(value);
}

}  // namespace

CliOptions parseCliOptions(const std::vector<std::string>& args) {
    CliOptions opts;
    std::vector<std::string> positional;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
            continue;
        }
        if (arg.rfind("--", 0) != 0) {
            positional.push_back(arg);
            continue;
        }
        if (i + 1 >= args.size()) throw std::invalid_argument(arg + ": missing value");
        const std::string& value = args[++i];
        if (arg == "--sigma") {
            opts.sigma = parseDouble(arg, value);
        } else if (arg == "--p") {
            opts.p = parseDouble(arg, value);
        } else if (arg == "--flip-mode") {
            opts.flipMode = parseFlipMode(value);
        } else if (arg == "--seed") {
            opts.seed = parseSeed(value);
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }

    // Same validation as the noise models, so errors appear before any file is read.
    static_cast<void>(ResistanceVariation{opts.sigma});
    static_cast<void>(BitFlip{opts.p, opts.flipMode});

    if (positional.size() > 3) throw std::invalid_argument("too many positional arguments");
    if (positional.size() > 0) opts.layer1Path = positional[0];
    if (positional.size() > 1) opts.layer2Path = positional[1];
    if (positional.size() > 2) opts.testDir = positional[2] + "/";
    return opts;
}

std::string cliUsage(const std::string& program) {
    return "Usage: " + program +
           " [layer1.csv] [layer2.csv] [test_dir]"
           " [--sigma S] [--p P] [--flip-mode cell|weight] [--seed N]\n"
           "  --sigma S      lognormal resistance variation (std of ln R), S >= 0, default 0\n"
           "  --p P          bit-flip probability, 0 <= P <= 1, default 0\n"
           "  --flip-mode M  cell (each MTJ device flips independently, default)\n"
           "                 or weight (both devices of a weight flip together)\n"
           "  --seed N       random seed, default 1\n"
           "With sigma = 0 and p = 0 (the default) noise is off.\n";
}
