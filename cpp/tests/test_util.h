// Helpers shared by the C++ tests.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"
#include "nn/evaluator.h"
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

// Wraps an evaluator for the driver tests: records the size of every call, keeps a
// copy of each call's planes, and throws on the call indices in `failAt` (1-based,
// counting failed calls too). Safe to call from one thread at a time (the drivers'
// evaluation thread); the counters are read after the run.
class RecordingEvaluator : public NNEvaluator {
 public:
  explicit RecordingEvaluator(NNEvaluator& inner) : inner_(inner) {}
  int boardSize() const override { return inner_.boardSize(); }
  const std::string& modelId() const override { return inner_.modelId(); }
  void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override {
    std::lock_guard<std::mutex> lk(m_);
    const int planeBytes = 17 * inner_.boardSize() * inner_.boardSize();
    std::vector<uint8_t> planes;
    for (const NNInput& i : in) planes.insert(planes.end(), i.planes, i.planes + planeBytes);
    calls_.push_back(std::move(planes));
    sizes_.push_back(static_cast<int>(in.size()));
    if (failAt.count(static_cast<int>(calls_.size()))) throw std::runtime_error("simulated evaluator failure");
    inner_.evaluate(in, out);
    positions_ += static_cast<int>(in.size());
  }
  std::set<int> failAt;
  int calls() const { return static_cast<int>(calls_.size()); }
  int positions() const { return positions_; }  // successfully evaluated
  int maxBatch() const {
    int m = 0;
    for (int s : sizes_) m = s > m ? s : m;
    return m;
  }
  const std::vector<int>& sizes() const { return sizes_; }
  const std::vector<uint8_t>& planesOfCall(int call) const { return calls_[static_cast<size_t>(call - 1)]; }

 private:
  NNEvaluator& inner_;
  std::mutex m_;
  std::vector<std::vector<uint8_t>> calls_;
  std::vector<int> sizes_;
  int positions_ = 0;
};

}  // namespace mango::test
