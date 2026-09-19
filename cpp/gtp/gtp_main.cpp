// mango_gtp: GTP engine.
//   mango_gtp [--size N] [--komi K] [--seed S]                         random player
//   mango_gtp --model <dir> [--sims N] [--device auto|cuda|mps|cpu] [--fp32]
//             [--config cfg.json] [--allow-komi-mismatch]              MCTS player (needs LibTorch)
// Option parsing and the size/komi precedence live in gtp_options.cpp (unit-tested).
#include <iostream>
#include <memory>
#include <string>

#include "core/config.h"
#include "gtp/gtp.h"
#include "gtp/gtp_options.h"
#include "gtp/mcts_player.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

int main(int argc, char** argv) {
  mango::GtpOptions opt;
  std::string error;
  if (!mango::GtpOptions::parse(argc, argv, &opt, &error)) {
    std::cerr << error << "\n";
    return 2;
  }
  if (opt.help) {
    std::cout << "usage: mango_gtp [--size N] [--komi K] [--seed S] [--model DIR --sims N --device D --fp32 --config CFG "
                 "--allow-komi-mismatch]\n";
    return 0;
  }
  try {
    const mango::Config base =
        opt.configPath.empty() ? mango::Config::fromJsonText("{}") : mango::Config::load(opt.configPath);

    if (opt.model.empty()) {
      mango::GtpSetup s = mango::resolveGtpSetup(opt, base, nullptr);
      mango::GtpEngine engine(std::make_unique<mango::RandomPlayer>(opt.seed), s.config.board.size, s.config.board.komi,
                              s.modelKomi);
      return engine.run(std::cin, std::cout);
    }
#ifdef MANGO_WITH_TORCH
    mango::TorchEvaluator::Options o;
    o.device = opt.device;
    o.fp16 = !opt.fp32;
    auto ev = std::make_unique<mango::TorchEvaluator>(opt.model, o);
    mango::GtpSetup s = mango::resolveGtpSetup(opt, base, &ev->meta());
    mango::SearchParams params =
        mango::SearchParams::fromConfig(s.config.search, s.config.board.size, /*selfplay=*/false);
    std::cerr << "mango_gtp: model " << ev->modelId() << " on " << ev->deviceName()
              << (ev->isHalf() ? " (fp16)" : " (fp32)") << ", " << params.simulations << " simulations/move\n";
    auto player = std::make_unique<mango::MctsPlayer>(*ev, params, opt.seed, "mango " + ev->modelId());
    mango::GtpEngine engine(std::move(player), s.config.board.size, s.config.board.komi, s.modelKomi);
    return engine.run(std::cin, std::cout);
#else
    std::cerr << "mango_gtp: built without LibTorch (MANGO_WITH_TORCH=OFF); --model is unavailable\n";
    return 1;
#endif
  } catch (const std::exception& e) {
    std::cerr << "mango_gtp: " << e.what() << "\n";
    return 1;
  }
}
