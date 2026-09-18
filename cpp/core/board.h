// Go position with Tromp-Taylor scoring, suicide prohibited, positional superko.
// Trivially copyable, no heap; keeps the last 8 stone configurations (snapshots)
// so the AGZ feature planes can be encoded without replaying the game.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/history.h"
#include "core/pointset.h"
#include "core/types.h"

namespace mango {

struct Snapshot {
  PointSet stones[2];  // [0] = black, [1] = white, indexed by unbordered point
  bool operator==(const Snapshot& o) const { return stones[0] == o.stones[0] && stones[1] == o.stones[1]; }
};

class Board {
 public:
  // moveCap < 0 selects the default 2*n*n.
  Board(int n, float komi, int moveCap = -1);

  // Build a position from a diagram for tests: rows of '.', 'X' (black), 'O' (white);
  // whitespace ignored; the number of rows is the board size. The resulting board has
  // moveCount 0 and this position as its only snapshot.
  static Board fromString(const std::string& diagram, float komi, Color toMove = Color::Black, int moveCap = -1);

  int size() const { return n_; }
  int numPoints() const { return n_ * n_; }
  float komi() const { return komi_; }
  int moveCap() const { return moveCap_; }

  Color at(int loc) const { return cells_[loc]; }
  Color get(int r, int c) const { return cells_[locOf(r, c)]; }
  Color atPoint(int p) const { return cells_[locOfPoint(p, n_)]; }
  Color toMove() const { return toMove_; }
  void setToMove(Color c) { toMove_ = c; }  // GTP: play for a colour out of turn
  int moveCount() const { return moveCount_; }
  int consecutivePasses() const { return passes_; }
  Move lastMove() const { return lastMove_; }
  uint64_t hash() const { return hash_; }  // stones only

  bool gameOver() const { return passes_ >= 2 || moveCount_ >= moveCap_; }

  // Legality for the side to move. Pass is always legal. Occupied or suicide -> false.
  // Superko: the resulting configuration must not be in hist.
  bool isLegal(Move m, const HashHistory& hist) const;
  // Occupied or suicide -> false. Otherwise *hashAfter = hash of the resulting position
  // (captures included), computed incrementally without modifying the board.
  bool tryMoveHash(Move m, uint64_t* hashAfter) const;
  // All legal moves for the side to move, pass last.
  void legalMoves(const HashHistory& hist, std::vector<Move>& out) const;

  // Plays a move for the side to move. Must be legal (checked in debug builds only
  // for occupancy/suicide; superko is the caller's responsibility).
  void play(Move m);

  // Area score: black - white - komi. Dead stones are not removed.
  float score() const;
  // owner[p] = +1 black, -1 white, 0 neutral (dame), for every point.
  void areaOwnership(int8_t* owner) const;

  int liberties(int p) const;  // 0 for empty points
  int chainSize(int p) const;  // 0 for empty points

  // Snapshot k steps back (0 = current). All-zero before the start of the game.
  const Snapshot& snapshot(int stepsBack) const;
  int snapshotCount() const { return snapCount_; }

  Board applySymmetry(int sym) const;  // transforms cells and all snapshots

  std::string toString() const;
  bool operator==(const Board& o) const;
  bool operator!=(const Board& o) const { return !(*this == o); }

 private:
  void placeStone(int loc, Color c);
  int mergeChains(int rootA, int rootB);  // returns surviving root
  void removeChain(int root);
  void rebuildChains();
  void pushSnapshot();
  void fillSnapshot(Snapshot& s) const;
  void setCellsFromSnapshot(const Snapshot& s);

  int n_;
  float komi_;
  int moveCap_;
  Color cells_[kMaxCells];
  int16_t chainId_[kMaxCells];   // root loc for stones, -1 otherwise
  int16_t next_[kMaxCells];      // circular list of the stones of a chain
  int16_t chainSize_[kMaxCells]; // valid at chain roots
  PointSet libs_[kMaxCells];     // valid at chain roots (unbordered point bits)
  Snapshot snaps_[kHistoryLen];
  int snapHead_;                 // index of the most recent snapshot
  int snapCount_;                // number of valid snapshots (<= kHistoryLen)
  uint64_t hash_;
  Color toMove_;
  int moveCount_;
  int passes_;
  Move lastMove_;
};

}  // namespace mango
