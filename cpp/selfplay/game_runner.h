// One self-play game with the search (docs/DESIGN.md sections 5.5, 5.4.6, 5.4.7):
// root noise, temperature moves, move cap, optional resignation, and the per-move
// data the chunk stores. `SelfplayGame` is a state machine used by both the
// sequential runner (playSelfplayGame) and the batched driver (batch_runner.h), so
// the two produce identical records for the same seed (DESIGN 5.4.5, K = 1).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/board.h"
#include "core/history.h"
#include "core/sgf.h"
#include "nn/evaluator.h"
#include "search/mcts.h"
#include "search/search_params.h"
#include "selfplay/chunk.h"

namespace mango {

struct SelfplayGameOptions {
  int boardSize = 9;
  float komi = 7.5f;
  int moveCap = -1;           // <0: 2*n*n
  SearchParams params;        // self-play params (noise on, temperature moves, resign threshold)
  uint64_t gameSeed = 0;      // drives noise, temperature sampling, symmetries
  bool noResignGame = false;  // play out even if resignation is enabled
  bool storeFinalOwnership = false;  // record_extras bit 0
  bool storeSearchKind = false;      // record_extras bit 1 (always 1 in this driver: full searches)
};

struct SelfplayGameResult {
  GameRecord record;
  SgfGame sgf;
  int evaluations = 0;  // positions sent to the network
};

class SelfplayGame {
 public:
  SelfplayGame(const SelfplayGameOptions& opt, std::string modelId);

  bool finished() const { return finished_; }
  const Board& board() const { return board_; }
  SearchTree& tree() { return *tree_; }

  // Ends the current move once the tree's budget is exhausted: records the root
  // statistics, chooses the move (or resigns), plays it and advances the tree.
  // Returns true when the game is over afterwards.
  bool finishMove();
  // The completed record (valid once finished()).
  SelfplayGameResult takeResult(int evaluations);

 private:
  void snapshot();

  SelfplayGameOptions opt_;
  std::string modelId_;
  Board board_;
  GameHistory hist_;
  std::unique_ptr<SearchTree> tree_;
  GameRecord record_;
  bool finished_ = false;
  bool resigned_ = false;
  Color resigner_ = Color::Empty;
};

// Plays a complete game from the empty board with the sequential search.
SelfplayGameResult playSelfplayGame(NNEvaluator& ev, const SelfplayGameOptions& opt, const std::string& modelId);

// Whether a game with this seed is a no-resign game under `fraction` (deterministic).
bool isNoResignGame(uint64_t gameSeed, float fraction);

}  // namespace mango
