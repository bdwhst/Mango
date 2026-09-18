// Slow, obviously-correct reference implementation of the rules, used only to
// fuzz the real Board. Full-board copies, flood fills, list of past positions.
#pragma once

#include <cstdint>
#include <vector>

#include "core/types.h"

namespace mango::test {

class RefBoard {
 public:
  RefBoard(int n, float komi, int moveCap = -1);

  int n;
  float komi;
  int moveCap;
  std::vector<Color> cells;                  // n*n, current position
  Color toMove;
  int passes;
  int moveCount;
  std::vector<std::vector<Color>> positions;  // every position so far, positions[0] = initial

  bool isLegal(Move m) const;  // occupancy, suicide, positional superko (pass always legal)
  void play(Move m);
  bool gameOver() const { return passes >= 2 || moveCount >= moveCap; }
  int liberties(int p) const;   // 0 for empty
  int chainSize(int p) const;
  float score() const;          // black - white - komi, area
  std::vector<uint8_t> features() const;  // 17 planes, AGZ layout
  uint64_t hash() const;        // Zobrist over stones, same keys as Board

  // Resulting position if m were played (captures applied). False if occupied/suicide.
  bool simulate(Move m, std::vector<Color>& out) const;

 private:
  void chainOf(const std::vector<Color>& cells, int p, std::vector<int>& chain, std::vector<int>& libs) const;
};

}  // namespace mango::test
