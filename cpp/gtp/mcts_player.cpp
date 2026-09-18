#include "gtp/mcts_player.h"

#include <stdexcept>

namespace mango {

MctsPlayer::MctsPlayer(NNEvaluator& ev, const SearchParams& params, uint64_t seed, std::string name)
    : ev_(ev), params_(params), seed_(seed), name_(std::move(name)) {}

void MctsPlayer::reset() {
  tree_.reset();
  syncedBoard_.reset();
  syncedMoves_ = 0;
}

void MctsPlayer::requireBoardSize(const Board& board) const {
  if (board.size() != ev_.boardSize())
    throw std::runtime_error("board size " + std::to_string(board.size()) + " does not match the model's " +
                             std::to_string(ev_.boardSize()));
}

void MctsPlayer::syncTree(const Board& board, const GameHistory& hist) {
  requireBoardSize(board);
  bool reused = false;
  if (tree_ && syncedBoard_ && hist.size() >= syncedMoves_ && hist.hashes.size() > syncedMoves_ &&
      hist.hashes[syncedMoves_] == syncedBoard_->hash()) {
    // Replay the appended moves on the synced board; each must reproduce the recorded
    // hash (a GTP "play" for the wrong colour or an undo breaks the prefix).
    Board b = *syncedBoard_;
    bool ok = true;
    for (size_t i = syncedMoves_; i < hist.size() && ok; ++i) {
      Move m = hist.moves[i];
      if (m != kPass && (m < 0 || m >= b.numPoints() || b.atPoint(m) != Color::Empty)) ok = false;
      if (ok) {
        b.play(m);
        ok = (b.hash() == hist.hashes[i + 1]);
      }
    }
    if (ok && b.hash() == board.hash() && b.toMove() == board.toMove() && b.moveCount() == board.moveCount()) {
      Board step = *syncedBoard_;
      for (size_t i = syncedMoves_; i < hist.size(); ++i) {
        step.play(hist.moves[i]);
        tree_->advance(hist.moves[i], step, hist);
      }
      reused = true;
    }
  }
  if (!reused) {
    if (!tree_) tree_ = std::make_unique<SearchTree>(params_, board.size(), seed_);
    tree_->newGame(board, hist);
  }
  syncedBoard_ = std::make_unique<Board>(board);
  syncedMoves_ = hist.size();
}

Move MctsPlayer::genmove(const Board& board, const GameHistory& hist) {
  requireBoardSize(board);
  if (board.gameOver()) return kPass;
  syncTree(board, hist);
  tree_->runSequential(ev_);
  return tree_->selectMove(board.moveCount());
  // The tree is advanced by the next syncTree() once the GTP layer has played the move.
}

std::string MctsPlayer::analyze(const Board& board, const GameHistory& hist) {
  requireBoardSize(board);
  if (board.gameOver()) return "game over";
  syncTree(board, hist);
  tree_->runSequential(ev_);
  return tree_->analyzeString(10);
}

}  // namespace mango
