// Match between two players (docs/DESIGN.md section 5.8): independent evaluators and
// trees, an opening set played as colour-swapped pairs, pair scores and a bootstrap
// interval. Evaluator-agnostic so tests can use FakeEvaluator.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/types.h"
#include "nn/evaluator.h"
#include "search/search_params.h"
#include "selfplay/chunk.h"

namespace mango {

struct Opening {
  std::vector<Move> moves;
};

// `count` distinct openings of `k` uniformly random legal moves each (seeded). Fewer
// are returned if the board cannot provide that many distinct ones.
std::vector<Opening> generateRandomOpenings(int n, float komi, int moveCap, int count, int k, uint64_t seed);

// Opening set <-> JSON (an array of move arrays; pass is written as n*n, the same
// encoding as the chunk format and the match report), the ladder's fixed opening
// file (DESIGN 6.6). `openingsFromJson` validates legality by replay and throws
// std::invalid_argument on a malformed or illegal opening.
std::string openingsToJson(const std::vector<Opening>& openings, int n);
std::vector<Opening> openingsFromJson(const std::string& text, int n, float komi, int moveCap);
// Move list in the JSON encoding (pass = n*n).
std::vector<int> movesToJson(const std::vector<Move>& moves, int n);

struct MatchPlayer {
  NNEvaluator* ev = nullptr;
  SearchParams params;  // eval params: no noise, tau -> 0, fixed simulations
  std::string name;
  // The ladder's random anchor (DESIGN 6.6): uniform over the legal board moves,
  // pass only when no board move is legal. No evaluator, no tree.
  bool random = false;
};

struct MatchGame {
  int pair = 0;
  bool aIsBlack = true;
  std::vector<Move> moves;  // opening included
  int8_t result = 0;        // +1 black, -1 white, 0 draw
  float score = 0.0f;
  Termination termination = Termination::TwoPasses;
  float scoreForA() const { return result == 0 ? 0.5f : ((result > 0) == aIsBlack ? 1.0f : 0.0f); }
};

struct MatchReport {
  std::string nameA, nameB;
  int boardSize = 0;
  float komi = 0.0f;
  int simulations = 0;
  uint64_t seed = 0;
  std::vector<Opening> openings;
  std::vector<MatchGame> games;   // 2 per opening: A black first, then A white
  std::vector<float> pairScores;  // mean of A's two game scores per opening
  double meanPairScore = 0.0;
  double ciLow = 0.0, ciHigh = 0.0;  // 95% bootstrap percentile interval over pairs
  int uniqueTrajectories = 0;
  int winsA = 0, lossesA = 0, drawsA = 0;
};

// Plays every opening twice with colours swapped. Game seeds derive from `seed`.
// `gamesInFlight` games run concurrently with one evaluator call per side per step
// (K = 1 per game, DESIGN 5.4.5): the result is identical for every value >= 1 with a
// deterministic evaluator; 0 means all games at once. `threads` > 1 (DESIGN 5.5.1, M4a)
// shares the game slots over T search threads with one evaluation thread that owns
// both evaluators and never mixes the two sides in one forward (`maxBatch` caps the
// requests it takes per step, <= 0: all in flight); threads == 1 is the first-version
// driver, unchanged. Under a deterministic evaluator the result does not depend on
// `threads` or `maxBatch`.
MatchReport playMatch(MatchPlayer a, MatchPlayer b, int n, float komi, int moveCap, const std::vector<Opening>& openings,
                      uint64_t seed, int gamesInFlight = 0, int threads = 1, int maxBatch = 0);

// Percentile bootstrap (2.5%, 97.5%) of the mean over `resamples` seeded resamples.
std::pair<double, double> bootstrapMeanInterval(const std::vector<float>& values, int resamples, uint64_t seed);

std::string matchReportToJson(const MatchReport& r);

}  // namespace mango
