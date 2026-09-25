// Batched self-play driver (docs/DESIGN.md section 5.5, 5.4.5): G games in flight,
// K = 1 leaf per game per step, one evaluator call per step. Each game issues exactly
// the sequence of collect/commit calls that SearchTree::runSequential would, so with a
// deterministic evaluator every game equals its standalone sequential run.
//
// With threads > 1 (DESIGN 5.5.1, M4a) the G slots are shared out over T search
// threads (thread t owns slots t, t+T, ...), which submit their pending leaves as rounds
// to one evaluation thread through an EvalQueue. threads == 1 is the first-version
// driver, unchanged.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "nn/evaluator.h"
#include "selfplay/game_runner.h"

namespace mango {

struct BatchStats {
  int games = 0;
  uint64_t positions = 0;
  uint64_t evaluations = 0;   // positions sent to the network
  uint64_t batches = 0;       // evaluator calls
  uint64_t retries = 0;       // evaluator failures retried
  double seconds = 0.0;       // wall clock of run()
  double evalSeconds = 0.0;   // inside evaluate()
  double avgBatch() const { return batches ? static_cast<double>(evaluations) / static_cast<double>(batches) : 0.0; }
};

class BatchedSelfplay {
 public:
  // `makeOptions(gameIndex)` supplies the options of the gameIndex-th game started
  // (seed, no-resign tag); `onGameDone` receives finished games in completion order.
  // With threads > 1 both callbacks are serialised (one mutex) and called from the
  // search threads.
  using OptionsFn = std::function<SelfplayGameOptions(int gameIndex)>;
  using DoneFn = std::function<void(int gameIndex, SelfplayGameResult&&)>;

  // threads: search threads (1 = the first-version driver); maxBatch: cap on the
  // requests per forward of the evaluation thread (<= 0: G), threads > 1 only.
  BatchedSelfplay(NNEvaluator& ev, int gamesInFlight, std::string modelId, int threads = 1, int maxBatch = 0);

  // Plays `totalGames` games and returns the statistics.
  BatchStats run(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone);

  int threads() const { return threads_; }
  int maxBatch() const { return maxBatch_; }

 private:
  struct Slot {
    std::unique_ptr<SelfplayGame> game;
    PendingLeaf leaf;
    int gameIndex = -1;
    int evaluations = 0;
    bool pending = false;  // leaf awaits the batch result
  };
  // Runs one game's collect loop until it has a pending leaf, its move budget is
  // exhausted (move finished here), or the game ends. Returns true if a leaf is pending.
  bool collectForSlot(Slot& s, BatchStats& stats, const DoneFn& onGameDone);
  // Starts the next game in the slot; games that are over before their first move
  // (move cap 0) are reported through `onGameDone` immediately. False when none is left.
  bool startGame(Slot& s, int& nextGameIndex, int totalGames, const OptionsFn& makeOptions, BatchStats& stats,
                 const DoneFn& onGameDone);
  BatchStats runSingleThread(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone);
  BatchStats runThreaded(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone);

  NNEvaluator& ev_;
  int G_;
  std::string modelId_;
  int threads_;
  int maxBatch_;
};

}  // namespace mango
