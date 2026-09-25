// mango_match: colour-swapped pairs between two model directories (DESIGN 5.8).
//   mango_match --a DIR|random --b DIR|random --config CFG --pairs N [--sims S] [--seed S]
//               [--openings K] [--openings-file F] [--out report.json] [--device D] [--fp32]
//               [--games-in-flight G]
//   mango_match --write-openings F --config CFG --pairs N [--openings K] [--seed S]
// "random" is the ladder's uniform-random legal-move anchor (DESIGN 6.6). With
// --openings-file the opening set is read from F (written by --write-openings or taken
// from a report) instead of being generated from the seed. Prints the JSON report to
// stdout (and to --out when given).
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "core/config.h"
#include "match/match.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

int main(int argc, char** argv) {
  std::string a, b, configPath, outPath, openingsFile, writeOpenings, device = "auto";
  int pairs = -1, sims = -1, openingMoves = -1, gamesInFlight = 32;
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
    else if (arg == "--openings-file") openingsFile = next();
    else if (arg == "--write-openings") writeOpenings = next();
    else if (arg == "--games-in-flight") gamesInFlight = std::atoi(next().c_str());
    else if (arg == "--device") device = next();
    else if (arg == "--fp32") fp32 = true;
    else {
      std::cerr << "unknown option " << arg << "\n";
      return 2;
    }
  }
  if (configPath.empty() || (writeOpenings.empty() && (a.empty() || b.empty()))) {
    std::cerr << "usage: mango_match --a DIR|random --b DIR|random --config CFG --pairs N [options]\n"
                 "       mango_match --write-openings F --config CFG --pairs N [--openings K] [--seed S]\n";
    return 2;
  }
  try {
    mango::Config cfg = mango::Config::load(configPath);
    if (pairs > 0) cfg.eval.pairs = pairs;
    if (sims > 0) cfg.search.evalSimulations = sims;
    if (openingMoves >= 0) cfg.eval.openingMoves = openingMoves;
    const int n = cfg.board.size;
    if (!writeOpenings.empty()) {
      auto openings = mango::generateRandomOpenings(n, cfg.board.komi, cfg.board.effectiveMoveCap(), cfg.eval.pairs,
                                                    cfg.eval.openingMoves, seed);
      std::ofstream f(writeOpenings + ".tmp");
      f << mango::openingsToJson(openings, n);
      f.close();
      std::remove(writeOpenings.c_str());
      if (std::rename((writeOpenings + ".tmp").c_str(), writeOpenings.c_str()) != 0)
        throw std::runtime_error("cannot write " + writeOpenings);
      std::cerr << "mango_match: wrote " << openings.size() << " openings to " << writeOpenings << "\n";
      return 0;
    }
    std::vector<mango::Opening> openings;
    if (!openingsFile.empty()) {
      std::ifstream f(openingsFile);
      if (!f) throw std::runtime_error("cannot read " + openingsFile);
      std::stringstream ss;
      ss << f.rdbuf();
      openings = mango::openingsFromJson(ss.str(), n, cfg.board.komi, cfg.board.effectiveMoveCap());
      if (pairs > 0 && static_cast<size_t>(pairs) < openings.size()) openings.resize(static_cast<size_t>(pairs));
    } else {
      openings = mango::generateRandomOpenings(n, cfg.board.komi, cfg.board.effectiveMoveCap(), cfg.eval.pairs,
                                               cfg.eval.openingMoves, seed);
    }
    const mango::SearchParams evalParams = mango::SearchParams::fromConfig(cfg.search, n, false);
    mango::MatchPlayer pa{nullptr, evalParams, "random", true};
    mango::MatchPlayer pb{nullptr, evalParams, "random", true};
    std::string deviceName = "cpu";
#ifdef MANGO_WITH_TORCH
    std::unique_ptr<mango::TorchEvaluator> evA, evB;
    mango::TorchEvaluator::Options o;
    o.device = device;
    o.fp16 = !fp32;
    if (a != "random") {
      evA = std::make_unique<mango::TorchEvaluator>(a, o);
      evA->meta().validate(cfg.board);
      pa = mango::MatchPlayer{evA.get(), evalParams, evA->modelId(), false};
      deviceName = evA->deviceName();
    }
    if (b != "random") {
      evB = std::make_unique<mango::TorchEvaluator>(b, o);
      evB->meta().validate(cfg.board);
      pb = mango::MatchPlayer{evB.get(), evalParams, evB->modelId(), false};
      deviceName = evB->deviceName();
    }
#else
    if (a != "random" || b != "random") {
      std::cerr << "mango_match: built without LibTorch (only random vs random is available)\n";
      return 1;
    }
#endif
    std::cerr << "mango_match: " << pa.name << " vs " << pb.name << ", " << openings.size() << " pairs, "
              << evalParams.simulations << " sims/move on " << deviceName << "\n";
    mango::MatchReport r = mango::playMatch(pa, pb, n, cfg.board.komi, cfg.board.effectiveMoveCap(), openings, seed, gamesInFlight);
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
}
