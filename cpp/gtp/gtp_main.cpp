// mango_gtp: GTP engine.
//   mango_gtp [--size N] [--komi K] [--seed S]                         random player
//   mango_gtp --model <dir> [--sims N] [--device auto|cuda|mps|cpu] [--fp32]
//             [--config cfg.json] [--allow-komi-mismatch]              MCTS player (needs LibTorch)
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "core/config.h"
#include "gtp/gtp.h"
#include "gtp/mcts_player.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

int main(int argc, char** argv) {
  int size = -1;
  float komi = 7.5f;
  bool komiGiven = false;
  uint64_t seed = 1;
  std::string model, device = "auto", configPath;
  int sims = -1;
  bool fp32 = false, allowKomiMismatch = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--size") size = std::atoi(next("--size"));
    else if (a == "--komi") {
      komi = static_cast<float>(std::atof(next("--komi")));
      komiGiven = true;
    }
    else if (a == "--seed") seed = std::strtoull(next("--seed"), nullptr, 10);
    else if (a == "--model") model = next("--model");
    else if (a == "--sims") sims = std::atoi(next("--sims"));
    else if (a == "--device") device = next("--device");
    else if (a == "--config") configPath = next("--config");
    else if (a == "--fp32") fp32 = true;
    else if (a == "--allow-komi-mismatch") allowKomiMismatch = true;
    else if (a == "--help" || a == "-h") {
      std::cout << "usage: mango_gtp [--size N] [--komi K] [--seed S] [--model DIR --sims N --device D --fp32 --config CFG]\n";
      return 0;
    } else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }
  mango::Config cfg = configPath.empty() ? mango::Config::fromJsonText("{}") : mango::Config::load(configPath);
  if (size > 0) cfg.board.size = size;
  if (komiGiven) cfg.board.komi = komi;
  if (sims > 0) cfg.search.evalSimulations = sims;

  if (model.empty()) {
    mango::GtpEngine engine(std::make_unique<mango::RandomPlayer>(seed), cfg.board.size, cfg.board.komi);
    return engine.run(std::cin, std::cout);
  }
#ifdef MANGO_WITH_TORCH
  try {
    mango::TorchEvaluator::Options o;
    o.device = device;
    o.fp16 = !fp32;
    auto ev = std::make_unique<mango::TorchEvaluator>(model, o);
    if (size <= 0) cfg.board.size = ev->meta().boardSize;
    if (!komiGiven) cfg.board.komi = ev->meta().komi;
    ev->meta().validate(cfg.board, allowKomiMismatch);
    // With --allow-komi-mismatch the GTP "komi" command accepts any value (-1 = unrestricted).
    const float gtpModelKomi = allowKomiMismatch ? -1.0f : ev->meta().komi;
    mango::SearchParams params = mango::SearchParams::fromConfig(cfg.search, cfg.board.size, /*selfplay=*/false);
    std::cerr << "mango_gtp: model " << ev->modelId() << " on " << ev->deviceName() << (ev->isHalf() ? " (fp16)" : " (fp32)")
              << ", " << params.simulations << " simulations/move\n";
    auto player = std::make_unique<mango::MctsPlayer>(*ev, params, seed, "mango " + ev->modelId());
    mango::GtpEngine engine(std::move(player), cfg.board.size, cfg.board.komi, gtpModelKomi);
    return engine.run(std::cin, std::cout);
  } catch (const std::exception& e) {
    std::cerr << "mango_gtp: " << e.what() << "\n";
    return 1;
  }
#else
  std::cerr << "mango_gtp: built without LibTorch (MANGO_WITH_TORCH=OFF); --model is unavailable\n";
  return 1;
#endif
}
