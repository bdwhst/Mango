#include "gtp/gtp_options.h"

#include <cstdlib>

namespace mango {

bool GtpOptions::parse(int argc, const char* const* argv, GtpOptions* out, std::string* error) {
  GtpOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what, const char** value) -> bool {
      if (i + 1 >= argc) {
        *error = std::string("missing value for ") + what;
        return false;
      }
      *value = argv[++i];
      return true;
    };
    const char* v = nullptr;
    if (a == "--size") {
      if (!next("--size", &v)) return false;
      o.size = std::atoi(v);
    } else if (a == "--komi") {
      if (!next("--komi", &v)) return false;
      o.komi = static_cast<float>(std::atof(v));
    } else if (a == "--seed") {
      if (!next("--seed", &v)) return false;
      o.seed = std::strtoull(v, nullptr, 10);
    } else if (a == "--model") {
      if (!next("--model", &v)) return false;
      o.model = v;
    } else if (a == "--sims") {
      if (!next("--sims", &v)) return false;
      o.sims = std::atoi(v);
    } else if (a == "--device") {
      if (!next("--device", &v)) return false;
      o.device = v;
    } else if (a == "--config") {
      if (!next("--config", &v)) return false;
      o.configPath = v;
    } else if (a == "--fp32") {
      o.fp32 = true;
    } else if (a == "--allow-komi-mismatch") {
      o.allowKomiMismatch = true;
    } else if (a == "--help" || a == "-h") {
      o.help = true;
    } else {
      *error = "unknown option " + a;
      return false;
    }
  }
  *out = o;
  return true;
}

GtpSetup resolveGtpSetup(const GtpOptions& opt, const Config& base, const ModelMeta* meta) {
  GtpSetup s;
  s.config = base;
  if (opt.size > 0) s.config.board.size = opt.size;
  else if (meta) s.config.board.size = meta->boardSize;
  if (opt.komi) s.config.board.komi = *opt.komi;
  else if (meta) s.config.board.komi = meta->komi;
  if (opt.sims > 0) s.config.search.evalSimulations = opt.sims;
  if (meta) {
    meta->validate(s.config.board, opt.allowKomiMismatch);
    if (!opt.allowKomiMismatch) s.modelKomi = meta->komi;
  }
  return s;
}

}  // namespace mango
