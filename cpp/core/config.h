// Run configuration (docs/DESIGN.md section 8). Loaded from JSON; every field has a
// default so partial files are valid.
#pragma once

#include <cstdint>
#include <string>

namespace mango {

struct BoardConfig {
  int size = 9;
  float komi = 7.5f;
  int moveCap = -1;  // -1 -> 2*size*size
  int effectiveMoveCap() const { return moveCap < 0 ? 2 * size * size : moveCap; }
};

struct SearchConfig {
  int simulations = 200;           // self-play budget (new simulations per move)
  int evalSimulations = 200;       // match / GTP / ladder budget
  float cPuct = 1.5f;
  std::string fpu = "zero";        // "zero" (paper) | "parent"
  float dirichletAlpha = -1.0f;    // <0 -> 0.03 * 361 / n^2
  float dirichletEpsilon = 0.25f;
  int temperatureMoves = 8;
  bool searchSymmetry = true;
  bool treeReuse = true;
  bool budgetIncludesInherited = false;
  float resignThreshold = -1.0f;   // -1 -> disabled (r < -1 never)
  float noResignFraction = 0.10f;
  int nnCacheSize = 0;
  bool resolveTerminalMoves = true;  // DESIGN 5.4.10 (D18); false = the paper's plain search
  float effectiveDirichletAlpha(int n) const {
    return dirichletAlpha >= 0 ? dirichletAlpha : 0.03f * 361.0f / static_cast<float>(n * n);
  }
};

struct SelfplayConfig {
  int gamesPerIteration = 2000;
  int gamesInFlight = 128;   // G
  int leavesPerGame = 1;     // K
  int chunkGames = 256;
  bool saveSgf = true;
};

struct TrainingConfig {
  int resBlocks = 6;
  int filters = 64;
  int batchSize = 256;
  int maxStepsPerIteration = 1000;
  float samplesPerPosition = 0.25f;
  int windowGames = 20000;
  float holdoutFraction = 0.05f;
  float l2 = 1e-4f;
  bool l2AllParams = false;
  std::string storePi = "visits";  // "visits" | "temperature"
  float lr0 = 0.01f;
  int lrStep1 = 30000;
  int lrStep2 = 60000;
};

struct EvalConfig {
  int pairs = 200;
  float gateThreshold = 0.55f;
  int openingMoves = 3;
  int ladderEvery = 5;
  bool gating = true;
};

struct Config {
  BoardConfig board;
  SearchConfig search;
  SelfplayConfig selfplay;
  TrainingConfig training;
  EvalConfig eval;

  static Config load(const std::string& path);
  static Config fromJsonText(const std::string& text);
  std::string toJsonText() const;
  // SHA-256-style fingerprint (16 hex chars) of board+search+selfplay, see DESIGN 5.6.
  std::string fingerprint() const;
};

}  // namespace mango
