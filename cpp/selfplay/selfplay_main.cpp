// mango_selfplay: sequential self-play (M3a'): one game at a time, K = 1.
//   mango_selfplay --model DIR --config CFG --games N --out DIR [--chunk-prefix P]
//                  [--chunk-id-start K] [--seed S] [--sgf-dir DIR] [--device D] [--fp32]
//                  [--iteration I]
// Publishes chunks of config selfplay.chunk_games games (atomic rename) and prints a
// JSON summary line to stdout on completion.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "core/config.h"
#include "core/random.h"
#include "nlohmann/json.hpp"
#include "selfplay/chunk.h"
#include "selfplay/game_runner.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string model, configPath, outDir, sgfDir, device = "auto", chunkPrefix = "chunk_";
  int games = -1, iteration = 0;
  uint64_t chunkIdStart = 0, seed = 1;
  bool fp32 = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--config") configPath = next();
    else if (a == "--games") games = std::atoi(next().c_str());
    else if (a == "--out") outDir = next();
    else if (a == "--chunk-prefix") chunkPrefix = next();
    else if (a == "--chunk-id-start") chunkIdStart = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--sgf-dir") sgfDir = next();
    else if (a == "--device") device = next();
    else if (a == "--iteration") iteration = std::atoi(next().c_str());
    else if (a == "--fp32") fp32 = true;
    else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }
  if (model.empty() || configPath.empty() || outDir.empty() || games <= 0) {
    std::cerr << "usage: mango_selfplay --model DIR --config CFG --games N --out DIR [options]\n";
    return 2;
  }
#ifndef MANGO_WITH_TORCH
  std::cerr << "mango_selfplay: built without LibTorch\n";
  return 1;
#else
  try {
    const mango::Config cfg = mango::Config::load(configPath);
    mango::TorchEvaluator::Options o;
    o.device = device;
    o.fp16 = !fp32;
    mango::TorchEvaluator ev(model, o);
    ev.meta().validate(cfg.board, /*allowKomiMismatch=*/false);
    fs::create_directories(outDir);
    if (!sgfDir.empty()) fs::create_directories(sgfDir);

    mango::ChunkHeader header;
    header.boardSize = cfg.board.size;
    header.komi = cfg.board.komi;
    header.simulationsPerMove = static_cast<uint32_t>(cfg.search.simulations);
    header.modelId = ev.modelId();
    header.configFingerprint = cfg.fingerprint();
    header.runSeed = seed;
    header.moveCap = static_cast<uint16_t>(cfg.board.effectiveMoveCap());
    header.temperatureMoves = static_cast<uint16_t>(cfg.search.temperatureMoves);
    header.recordExtras = 0;  // AGZ profile: no optional fields (DESIGN 5.6)

    mango::SelfplayGameOptions opt;
    opt.boardSize = cfg.board.size;
    opt.komi = cfg.board.komi;
    opt.moveCap = cfg.board.effectiveMoveCap();
    opt.params = mango::SearchParams::fromConfig(cfg.search, cfg.board.size, /*selfplay=*/true);

    std::cerr << "mango_selfplay: model " << ev.modelId() << " on " << ev.deviceName() << (ev.isHalf() ? " (fp16)" : " (fp32)")
              << ", " << opt.params.simulations << " sims/move, " << games << " games\n";

    const int perChunk = std::max(1, cfg.selfplay.chunkGames);
    int played = 0;
    uint64_t positions = 0, evaluations = 0;
    std::map<int, int> terminations;
    int blackWins = 0;
    uint64_t chunkId = chunkIdStart;
    std::unique_ptr<mango::ChunkWriter> writer;
    std::string chunkPath;
    auto openChunk = [&]() {
      char name[64];
      std::snprintf(name, sizeof name, "%s%06llu.mgo", chunkPrefix.c_str(), static_cast<unsigned long long>(chunkId));
      chunkPath = (fs::path(outDir) / name).string();
      header.chunkId = chunkId;
      writer = std::make_unique<mango::ChunkWriter>(chunkPath, header);
    };
    std::vector<std::string> published;
    while (played < games) {
      if (!writer) openChunk();
      const int gameIndex = writer->games();
      opt.gameSeed = mango::deriveSeed(seed, chunkId, static_cast<uint64_t>(gameIndex));
      opt.noResignGame = mango::isNoResignGame(opt.gameSeed, cfg.search.noResignFraction);
      mango::SelfplayGameResult r = mango::playSelfplayGame(ev, opt, ev.modelId());
      writer->append(r.record);
      ++played;
      positions += static_cast<uint64_t>(r.record.T());
      evaluations += static_cast<uint64_t>(r.evaluations);
      terminations[static_cast<int>(r.record.termination)] += 1;
      blackWins += r.record.result > 0;
      if (!sgfDir.empty() && cfg.selfplay.saveSgf) {
        char sgfName[96];
        std::snprintf(sgfName, sizeof sgfName, "%s%06llu_%04d.sgf", chunkPrefix.c_str(), static_cast<unsigned long long>(chunkId),
                      gameIndex);
        std::ofstream f(fs::path(sgfDir) / sgfName);
        f << mango::writeSgf(r.sgf);
      }
      if (writer->games() >= perChunk || played == games) {
        published.push_back(writer->close());
        writer.reset();
        ++chunkId;
      }
    }
    nlohmann::json summary;
    summary["games"] = played;
    summary["positions"] = positions;
    summary["evaluations"] = evaluations;
    summary["avg_game_length"] = played ? static_cast<double>(positions) / played : 0.0;
    summary["black_wins"] = blackWins;
    summary["terminations"] = {{"two_passes", terminations[0]}, {"resign", terminations[1]}, {"move_cap", terminations[2]}};
    summary["chunks"] = published;
    summary["next_chunk_id"] = chunkId;
    summary["model_id"] = ev.modelId();
    summary["iteration"] = iteration;
    std::cout << summary.dump() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mango_selfplay: " << e.what() << "\n";
    return 1;
  }
#endif
}
