// mango_selfplay: self-play with G games in flight, K = 1 (DESIGN 5.5).
//   mango_selfplay --model DIR --config CFG --games N --out DIR [--chunk-prefix P]
//                  [--chunk-id-start K] [--seed S] [--sgf-dir DIR] [--device D] [--fp32]
//                  [--iteration I] [--games-in-flight G] [--resign-threshold T]
// --resign-threshold overrides search.resign_threshold (the pipeline passes the value
// selected per iteration, DESIGN 5.4.7; -1 disables resignation).
// Publishes chunks of config selfplay.chunk_games games (atomic rename) and prints a
// JSON summary line (games, positions, evaluations, timing) to stdout on completion.
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
#include "selfplay/batch_runner.h"
#include "selfplay/chunk.h"
#include "selfplay/game_runner.h"
#ifdef MANGO_WITH_TORCH
#include "nn/torch_evaluator.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string model, configPath, outDir, sgfDir, device = "auto", chunkPrefix = "chunk_";
  int games = -1, iteration = 0, gamesInFlight = -1;
  uint64_t chunkIdStart = 0, seed = 1;
  bool fp32 = false;
  bool haveResign = false;
  float resignThreshold = -1.0f;
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
    else if (a == "--games-in-flight") gamesInFlight = std::atoi(next().c_str());
    else if (a == "--resign-threshold") {
      resignThreshold = std::strtof(next().c_str(), nullptr);
      haveResign = true;
    }
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
    mango::Config cfg = mango::Config::load(configPath);
    if (haveResign) cfg.search.resignThreshold = resignThreshold;
    mango::TorchEvaluator::Options o;
    o.device = device;
    o.fp16 = !fp32;
    mango::TorchEvaluator ev(model, o);
    ev.meta().validate(cfg.board, /*allowKomiMismatch=*/false);
    fs::create_directories(outDir);
    if (!sgfDir.empty()) fs::create_directories(sgfDir);
    const int G = gamesInFlight > 0 ? gamesInFlight : cfg.selfplay.gamesInFlight;

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

    mango::SelfplayGameOptions base;
    base.boardSize = cfg.board.size;
    base.komi = cfg.board.komi;
    base.moveCap = cfg.board.effectiveMoveCap();
    base.params = mango::SearchParams::fromConfig(cfg.search, cfg.board.size, /*selfplay=*/true);

    std::cerr << "mango_selfplay: model " << ev.modelId() << " on " << ev.deviceName() << (ev.isHalf() ? " (fp16)" : " (fp32)")
              << ", " << base.params.simulations << " sims/move, " << games << " games, " << G << " in flight, resign "
              << (base.params.resignThreshold <= -1.0f ? std::string("off") : std::to_string(base.params.resignThreshold))
              << "\n";

    const int perChunk = std::max(1, cfg.selfplay.chunkGames);
    std::map<int, int> terminations;
    int blackWins = 0;
    uint64_t chunkId = chunkIdStart;
    std::unique_ptr<mango::ChunkWriter> writer;
    std::vector<std::string> published;
    auto openChunk = [&]() {
      char name[64];
      std::snprintf(name, sizeof name, "%s%06llu.mgo", chunkPrefix.c_str(), static_cast<unsigned long long>(chunkId));
      header.chunkId = chunkId;
      writer = std::make_unique<mango::ChunkWriter>((fs::path(outDir) / name).string(), header);
    };
    // Game seeds: hash(run seed, chunk id at start, start index within the run). Games
    // finish out of order, so the record's gameIndex (assigned by the writer at append
    // time) differs from the start index; the seed is stored in the record either way.
    auto makeOptions = [&](int startIndex) {
      mango::SelfplayGameOptions opt = base;
      opt.gameSeed = mango::deriveSeed(seed, chunkIdStart, static_cast<uint64_t>(startIndex));
      opt.noResignGame = mango::isNoResignGame(opt.gameSeed, cfg.search.noResignFraction);
      return opt;
    };
    int written = 0;
    auto onDone = [&](int startIndex, mango::SelfplayGameResult&& r) {
      if (!writer) openChunk();
      const int gameIndex = writer->games();
      writer->append(r.record);
      ++written;
      terminations[static_cast<int>(r.record.termination)] += 1;
      blackWins += r.record.result > 0;
      if (!sgfDir.empty() && cfg.selfplay.saveSgf) {
        char sgfName[96];
        std::snprintf(sgfName, sizeof sgfName, "%s%06llu_%04d.sgf", chunkPrefix.c_str(), static_cast<unsigned long long>(chunkId),
                      gameIndex);
        std::ofstream f(fs::path(sgfDir) / sgfName);
        f << mango::writeSgf(r.sgf);
      }
      (void)startIndex;
      if (writer->games() >= perChunk || written == games) {
        published.push_back(writer->close());
        writer.reset();
        ++chunkId;
      }
    };
    mango::BatchedSelfplay driver(ev, G, ev.modelId());
    const mango::BatchStats st = driver.run(games, makeOptions, onDone);
    if (writer) {  // cannot happen (closed on the last game), kept for safety
      published.push_back(writer->close());
      writer.reset();
      ++chunkId;
    }
    nlohmann::json summary;
    summary["games"] = st.games;
    summary["positions"] = st.positions;
    summary["evaluations"] = st.evaluations;
    summary["batches"] = st.batches;
    summary["avg_batch"] = st.avgBatch();
    summary["retries"] = st.retries;
    summary["seconds"] = st.seconds;
    summary["eval_seconds"] = st.evalSeconds;
    summary["evals_per_s"] = st.seconds > 0 ? static_cast<double>(st.evaluations) / st.seconds : 0.0;
    summary["positions_per_s"] = st.seconds > 0 ? static_cast<double>(st.positions) / st.seconds : 0.0;
    summary["games_in_flight"] = G;
    summary["avg_game_length"] = st.games ? static_cast<double>(st.positions) / st.games : 0.0;
    summary["black_wins"] = blackWins;
    summary["terminations"] = {{"two_passes", terminations[0]}, {"resign", terminations[1]}, {"move_cap", terminations[2]}};
    summary["resign_threshold"] = base.params.resignThreshold;
    summary["chunks"] = published;
    summary["next_chunk_id"] = chunkId;
    summary["model_id"] = ev.modelId();
    summary["device"] = ev.deviceName() + (ev.isHalf() ? " fp16" : " fp32");
    summary["iteration"] = iteration;
    std::cout << summary.dump() << "\n";
    std::cerr << "mango_selfplay: " << st.games << " games, " << st.positions << " positions in " << st.seconds << " s ("
              << summary["positions_per_s"].get<double>() << " pos/s, " << summary["evals_per_s"].get<double>()
              << " evals/s, avg batch " << st.avgBatch() << ", eval " << 100.0 * st.evalSeconds / std::max(st.seconds, 1e-9)
              << "% of wall)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mango_selfplay: " << e.what() << "\n";
    return 1;
  }
#endif
}
