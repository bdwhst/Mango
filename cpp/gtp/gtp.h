// GTP v2 front-end. The move generator is pluggable so the same engine serves the
// random player (M1) and the MCTS player (M3).
#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"

namespace mango {

class Player {
 public:
  virtual ~Player() = default;
  virtual std::string name() const = 0;
  // Chooses a move for `board.toMove()`.
  virtual Move genmove(const Board& board, const GameHistory& hist) = 0;
  // Called whenever the position is changed by something other than a forward move
  // (undo, clear_board, boardsize, komi): any retained search state must be discarded.
  virtual void reset() {}
  // Whether this player can play on an n x n board (a network is size-specific).
  virtual bool acceptsBoardSize(int) const { return true; }
  // Optional analysis output for the "mango-analyze" command.
  virtual std::string analyze(const Board&, const GameHistory&) { return ""; }
};

class RandomPlayer : public Player {
 public:
  explicit RandomPlayer(uint64_t seed) : rng_(seed) {}
  std::string name() const override { return "mango-random"; }
  Move genmove(const Board& board, const GameHistory& hist) override;

 private:
  Rng rng_;
  std::vector<Move> legal_;
};

class GtpEngine {
 public:
  GtpEngine(std::unique_ptr<Player> player, int size, float komi, float modelKomi = -1.0f);
  // Runs the command loop until "quit" or EOF. Returns 0.
  int run(std::istream& in, std::ostream& out);
  // Handles one command line; returns the full response (without trailing blank line)
  // and sets *quit when the command was "quit".
  std::string handle(const std::string& line, bool* quit);

  static std::string vertexToString(Move m, int n);
  static bool parseVertex(const std::string& s, int n, Move* out);
  const Board& board() const { return board_; }
  const GameHistory& history() const { return hist_; }

 private:
  void newGame(int size);
  void replayAll();

  // Plays `m` for colour `c` (GTP allows either colour at any time). Returns false
  // and leaves the position untouched if the move is illegal.
  bool playMove(Color c, Move m);

  std::unique_ptr<Player> player_;
  int size_;
  float komi_;
  float modelKomi_;  // <0: any komi accepted
  Board board_;
  GameHistory hist_;
  std::vector<Color> colors_;  // colour of each move in hist_ (GTP moves need not alternate)
};

}  // namespace mango
