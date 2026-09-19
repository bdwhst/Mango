// Command-line options of mango_gtp and the decision "which board size, komi and
// model-komi restriction does the engine run with". Kept out of main() so it is
// unit-testable (docs/DESIGN.md section 5.7 lists the precedence rules).
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/config.h"
#include "nn/model_meta.h"

namespace mango {

struct GtpOptions {
  int size = -1;                 // <=0: from config / model
  std::optional<float> komi;     // nullopt: from config / model (negative values are valid)
  uint64_t seed = 1;
  std::string model;             // empty: random player
  std::string device = "auto";
  std::string configPath;
  int sims = -1;                 // <=0: from config
  bool fp32 = false;
  bool allowKomiMismatch = false;
  bool help = false;

  // Parses argv[1..]. Returns false with *error set on an unknown option or a
  // missing value; `help` is set (and true returned) for --help / -h.
  static bool parse(int argc, const char* const* argv, GtpOptions* out, std::string* error);
};

struct GtpSetup {
  Config config;                     // board/search sections after all overrides
  std::optional<float> modelKomi;    // what the GTP "komi" command must match; nullopt = any
};

// Precedence: command line > model metadata > config file > defaults.
//   size:  --size, else (with a model) the model's size, else config.
//   komi:  --komi (any sign), else (with a model) the model's komi, else config.
//   sims:  --sims, else config.search.evalSimulations.
// With a model the result is validated against the model (ModelMeta::validate); a komi
// mismatch throws unless --allow-komi-mismatch, in which case the "komi" GTP command is
// also unrestricted. Without a model the "komi" command is unrestricted.
// `meta` may be null (random player).
GtpSetup resolveGtpSetup(const GtpOptions& opt, const Config& base, const ModelMeta* meta);

}  // namespace mango
