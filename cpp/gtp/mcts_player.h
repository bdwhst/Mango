// GTP player backed by the PUCT search and any NNEvaluator.
#pragma once

#include <memory>
#include <string>

#include "gtp/gtp.h"
#include "nn/evaluator.h"
#include "search/mcts.h"

namespace mango {

class MctsPlayer : public Player {
 public:
  // The evaluator is owned by the caller and must outlive the player.
  MctsPlayer(NNEvaluator& ev, const SearchParams& params, uint64_t seed, std::string name = "mango");
  std::string name() const override { return name_; }
  Move genmove(const Board& board, const GameHistory& hist) override;
  void reset() override;
  std::string analyze(const Board& board, const GameHistory& hist) override;
  bool acceptsBoardSize(int n) const override { return n == ev_.boardSize(); }

  // The current tree (nullptr before the first search); for tests and diagnostics.
  const SearchTree* tree() const { return tree_.get(); }

 private:
  // Brings the tree in line with (board, hist): advances through every move appended
  // since the last sync while the recorded prefix still matches, otherwise rebuilds.
  void syncTree(const Board& board, const GameHistory& hist);
  void requireBoardSize(const Board& board) const;

  NNEvaluator& ev_;
  SearchParams params_;
  uint64_t seed_;
  std::string name_;
  std::unique_ptr<SearchTree> tree_;
  std::unique_ptr<Board> syncedBoard_;  // position the tree's root corresponds to
  size_t syncedMoves_ = 0;              // hist.size() at the last sync
};

}  // namespace mango
