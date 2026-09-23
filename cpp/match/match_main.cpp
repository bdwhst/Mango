// mango_match: colour-swapped pairs between two model directories (DESIGN 5.8).
//   mango_match --a DIR --b DIR --config CFG --pairs N [--sims S] [--seed S]
//               [--openings K] [--out report.json] [--device D] [--fp32]
// Prints the JSON report to stdout (and to --out when given).
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "core/config.h"
#include "match/match.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

int main(int argc, char** argv) {
  std::string a, b, configPath, outPath, device = "auto";
  int pairs = -1, sims = -1, openingMoves = -1;
  uint64_t seed = 1;
  bool fp32 = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--a") a = next();
    else if (arg == "--b") b = next();
    else if (arg == "--config") configPath = next();
    else if (arg == "--pairs") pairs = std::atoi(next().c_str());
    else if (arg == "--sims") sims = std::atoi(next().c_str());
    else if (arg == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (arg == "--openings") openingMoves = std::atoi(next().c_str());
    else if (arg == "--out") outPath = next();
    else if (arg == "--device") device = next();
    else if (arg == "--fp32") fp32 = true;
    else {
      std::cerr << "unknown option " << arg << "\n";
      return 2;
    }
  }
  if (a.empty() || b.empty() || configPath.empty()) {
    std::cerr << "usage: mango_match --a DIR --b DIR --config CFG --pairs N [options]\n";
    return 2;
  }
#ifndef MANGO_WITH_TORCH
  std::cerr << "mango_match: built without LibTorch\n";
  return 1;
#else
  try {
    mango::Config cfg = mango::Config::load(configPath);
    if (pairs > 0) cfg.eval.pairs = pairs;
    if (sims > 0) cfg.search.evalSimulations = sims;
    if (openingMoves >= 0) cfg.eval.openingMoves = openingMoves;
    mango::TorchEvaluator::Options o;
    o.device = device;
    o.fp16 = !fp32;
    mango::TorchEvaluator evA(a, o);
    mango::TorchEvaluator evB(b, o);
    evA.meta().validate(cfg.board);
    evB.meta().validate(cfg.board);
    const int n = cfg.board.size;
    mango::MatchPlayer pa{&evA, mango::SearchParams::fromConfig(cfg.search, n, false), evA.modelId()};
    mango::MatchPlayer pb{&evB, mango::SearchParams::fromConfig(cfg.search, n, false), evB.modelId()};
    auto openings = mango::generateRandomOpenings(n, cfg.board.komi, cfg.board.effectiveMoveCap(), cfg.eval.pairs,
                                                  cfg.eval.openingMoves, seed);
    std::cerr << "mango_match: " << evA.modelId() << " vs " << evB.modelId() << ", " << openings.size() << " pairs, "
              << pa.params.simulations << " sims/move on " << evA.deviceName() << "\n";
    mango::MatchReport r = mango::playMatch(pa, pb, n, cfg.board.komi, cfg.board.effectiveMoveCap(), openings, seed);
    const std::string json = mango::matchReportToJson(r);
    if (!outPath.empty()) {
      std::ofstream f(outPath + ".tmp");
      f << json;
      f.close();
      std::remove(outPath.c_str());
      std::rename((outPath + ".tmp").c_str(), outPath.c_str());
    }
    std::cout << json << "\n";
    std::cerr << "mango_match: mean pair score " << r.meanPairScore << " [" << r.ciLow << ", " << r.ciHigh << "], unique "
              << r.uniqueTrajectories << "/" << r.games.size() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mango_match: " << e.what() << "\n";
    return 1;
  }
#endif
}
