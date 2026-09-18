// Helpers shared by the C++ tests.
#pragma once

#include <cstdlib>
#include <vector>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"
#include "tests/ref_board.h"

namespace mango::test {

// Plays one random game on both boards in lockstep (moves chosen from the real
// Board's legal list), calling `step(board, ref, hist)` before every move and once
// at the end. Pass is chosen with probability passProb when other moves exist.
template <typename Step>
void randomGame(int n, float komi, uint64_t seed, double passProb, Step step) {
  Board board(n, komi);
  RefBoard ref(n, komi);
  GameHistory hist;
  hist.reset(board.hash());
  Rng rng(seed);
  std::vector<Move> legal;
  while (!board.gameOver()) {
    step(board, ref, hist);
    board.legalMoves(HashHistory(hist), legal);
    Move m;
    if (legal.size() == 1 || rng.uniformReal() < passProb) m = kPass;
    else m = legal[rng.uniformInt(static_cast<uint32_t>(legal.size() - 1))];
    board.play(m);
    ref.play(m);
    hist.push(m, board.hash());
  }
  step(board, ref, hist);
}

inline int fuzzScale() {
  const char* s = std::getenv("MANGO_FUZZ_SCALE");
  return s ? std::atoi(s) : 1;
}

}  // namespace mango::test
