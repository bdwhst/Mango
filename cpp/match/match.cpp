#include "match/match.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <sstream>
#include <thread>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"
#include "core/zobrist.h"
#include "nlohmann/json.hpp"
#include "search/mcts.h"
#include "selfplay/eval_queue.h"

namespace mango {

std::vector<Opening> generateRandomOpenings(int n, float komi, int moveCap, int count, int k, uint64_t seed) {
  Rng rng(seed);
  std::vector<Opening> out;
  std::set<std::vector<Move>> seen;
  std::vector<Move> legal;
  int attempts = 0;
  while (static_cast<int>(out.size()) < count && attempts < count * 50 + 100) {
    ++attempts;
    Board b(n, komi, moveCap);
    GameHistory h;
    h.reset(b.hash());
    Opening o;
    bool ok = true;
    for (int i = 0; i < k; ++i) {
      b.legalMoves(HashHistory(h), legal);
      // Board moves only (pass is last); an opening never contains a pass.
      if (legal.size() <= 1) {
        ok = false;
        break;
      }
      Move m = legal[rng.uniformInt(static_cast<uint32_t>(legal.size() - 1))];
      o.moves.push_back(m);
      b.play(m);
      h.push(m, b.hash());
    }
    if (!ok || !seen.insert(o.moves).second) continue;
    out.push_back(std::move(o));
  }
  return out;
}

std::vector<int> movesToJson(const std::vector<Move>& moves, int n) {
  std::vector<int> out;
  out.reserve(moves.size());
  for (Move m : moves) out.push_back(m == kPass ? n * n : static_cast<int>(m));
  return out;
}

std::string openingsToJson(const std::vector<Opening>& openings, int n) {
  nlohmann::json ops = nlohmann::json::array();
  for (const Opening& o : openings) ops.push_back(movesToJson(o.moves, n));
  return ops.dump();
}

std::vector<Opening> openingsFromJson(const std::string& text, int n, float komi, int moveCap) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("openings: not JSON: ") + e.what());
  }
  if (!j.is_array()) throw std::invalid_argument("openings: expected an array of move arrays");
  std::vector<Opening> out;
  std::vector<Move> legal;
  for (size_t i = 0; i < j.size(); ++i) {
    if (!j[i].is_array()) throw std::invalid_argument("openings: entry " + std::to_string(i) + " is not an array");
    Opening o;
    Board b(n, komi, moveCap);
    GameHistory h;
    h.reset(b.hash());
    for (const auto& v : j[i]) {
      if (!v.is_number_integer()) throw std::invalid_argument("openings: entry " + std::to_string(i) + " has a non-integer move");
      const long long m = v.get<long long>();
      if (m < 0 || m > n * n) throw std::invalid_argument("openings: entry " + std::to_string(i) + " has a move out of range");
      const Move mv = m == n * n ? kPass : static_cast<Move>(m);
      if (!b.isLegal(mv, HashHistory(h))) throw std::invalid_argument("openings: entry " + std::to_string(i) + " is illegal");
      b.play(mv);
      h.push(mv, b.hash());
      o.moves.push_back(mv);
    }
    out.push_back(std::move(o));
  }
  return out;
}

namespace {

// One game in flight: both sides' trees follow the same board. A random side (the
// ladder anchor) has no tree; its moves come from its own seeded generator.
struct MatchSlot {
  int pair = 0;
  bool aIsBlack = true;
  Board board;
  GameHistory hist;
  std::unique_ptr<SearchTree> tb, tw;
  Rng rngB, rngW;
  MatchGame game;
  PendingLeaf leaf;
  bool active = false;
  std::vector<Move> legal;
  MatchSlot() : board(2, 0.0f), rngB(0), rngW(0) {}

  SearchTree* treePtrToMove() { return board.toMove() == Color::Black ? tb.get() : tw.get(); }
  SearchTree& treeToMove() { return *treePtrToMove(); }
  bool moverIsA() const { return (board.toMove() == Color::Black) == aIsBlack; }

  void start(int pairIdx, bool aBlack, MatchPlayer& black, MatchPlayer& white, int n, float komi, int moveCap,
             const Opening& opening, uint64_t seedBlack, uint64_t seedWhite) {
    pair = pairIdx;
    aIsBlack = aBlack;
    board = Board(n, komi, moveCap);
    hist.reset(board.hash());
    tb.reset();
    tw.reset();
    if (black.random) {
      rngB.reseed(seedBlack);
    } else {
      tb = std::make_unique<SearchTree>(black.params, n, seedBlack);
      tb->newGame(board, hist);
    }
    if (white.random) {
      rngW.reseed(seedWhite);
    } else {
      tw = std::make_unique<SearchTree>(white.params, n, seedWhite);
      tw->newGame(board, hist);
    }
    game = MatchGame();
    game.pair = pairIdx;
    game.aIsBlack = aBlack;
    for (Move m : opening.moves) apply(m);
    active = true;
    afterMove();
  }
  void apply(Move m) {
    board.play(m);
    hist.push(m, board.hash());
    if (tb) tb->advance(m, board, hist);
    if (tw) tw->advance(m, board, hist);
    game.moves.push_back(m);
  }
  // The new side to move starts its search (root noise is off in matches, so this only
  // matters for a reused root), exactly as runSequential would.
  void afterMove() {
    if (board.gameOver()) return;
    if (SearchTree* t = treePtrToMove()) t->prepareRoot();
  }
  Move randomMove() {
    Rng& rng = board.toMove() == Color::Black ? rngB : rngW;
    board.legalMoves(HashHistory(hist), legal);  // pass is last
    if (legal.size() == 1) return kPass;
    return legal[rng.uniformInt(static_cast<uint32_t>(legal.size() - 1))];
  }
  void finishGame() {
    game.score = board.score();
    game.result = game.score > 0 ? 1 : (game.score < 0 ? -1 : 0);
    game.termination = board.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
    active = false;
  }
  // Same call sequence as SearchTree::runSequential for the side to move (DESIGN 5.4.5).
  // Returns true with a pending leaf, false when the game ended.
  bool collect() {
    for (;;) {
      if (board.gameOver()) {
        finishGame();
        return false;
      }
      SearchTree* treePtr = treePtrToMove();
      if (treePtr == nullptr) {  // random side: no search
        apply(randomMove());
        afterMove();
        continue;
      }
      SearchTree& tree = *treePtr;
      if (!tree.rootExpanded()) {
        if (tree.collectLeaf(leaf) != CollectResult::Pending) throw std::logic_error("unexpected collect result at the root");
        return true;
      }
      if (tree.budgetExhausted()) {
        apply(tree.selectMove(board.moveCount()));
        afterMove();
        continue;
      }
      const CollectResult r = tree.collectLeaf(leaf);
      if (r == CollectResult::Pending) return true;
      if (r == CollectResult::Collision) throw std::logic_error("collision with K = 1");
    }
  }
};

uint64_t trajectoryHash(const std::vector<Move>& moves) {
  uint64_t h = 0x4D616E676F4D6174ull;
  for (Move m : moves) h = deriveSeed(h, static_cast<uint64_t>(static_cast<int64_t>(m) + 2));
  return h;
}

}  // namespace

std::pair<double, double> bootstrapMeanInterval(const std::vector<float>& values, int resamples, uint64_t seed) {
  if (values.empty()) return {0.0, 0.0};
  Rng rng(seed);
  std::vector<double> means;
  means.reserve(resamples);
  const uint32_t m = static_cast<uint32_t>(values.size());
  for (int r = 0; r < resamples; ++r) {
    double s = 0.0;
    for (uint32_t i = 0; i < m; ++i) s += values[rng.uniformInt(m)];
    means.push_back(s / m);
  }
  std::sort(means.begin(), means.end());
  auto pct = [&](double q) {
    const double pos = q * (means.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, means.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return means[lo] * (1.0 - frac) + means[hi] * frac;
  };
  return {pct(0.025), pct(0.975)};
}

namespace {

// Starts game `gi` (opening gi/2, A black when gi is even) in the slot.
void startMatchGame(MatchSlot& s, int gi, MatchPlayer& a, MatchPlayer& b, int n, float komi, int moveCap,
                    const std::vector<Opening>& openings, uint64_t seed) {
  const size_t i = static_cast<size_t>(gi / 2);
  const int side = gi % 2;
  const bool aIsBlack = side == 0;
  MatchPlayer& black = aIsBlack ? a : b;
  MatchPlayer& white = aIsBlack ? b : a;
  s.start(static_cast<int>(i), aIsBlack, black, white, n, komi, moveCap, openings[i], deriveSeed(seed, i * 4 + side * 2 + 0),
          deriveSeed(seed, i * 4 + side * 2 + 1));
}

// The first-version match driver (M3b): one thread, one evaluator call per side per step.
void playMatchSingleThread(MatchReport& r, MatchPlayer& a, MatchPlayer& b, int n, float komi, int moveCap,
                           const std::vector<Opening>& openings, uint64_t seed, int G) {
  const int totalGames = static_cast<int>(r.games.size());
  std::vector<MatchSlot> slots(static_cast<size_t>(G));
  int nextGame = 0;
  auto startNext = [&](MatchSlot& s) {
    if (nextGame >= totalGames) return false;
    startMatchGame(s, nextGame++, a, b, n, komi, moveCap, openings, seed);
    return true;
  };
  int active = 0;
  for (MatchSlot& s : slots)
    if (startNext(s)) ++active;
  std::vector<MatchSlot*> batchA, batchB;
  std::vector<NNInput> in;
  std::vector<NNOutput> out;
  auto flush = [&](std::vector<MatchSlot*>& batch, NNEvaluator* ev) {
    if (batch.empty()) return;
    if (ev == nullptr) throw std::logic_error("a random player has no pending leaves");
    in.clear();
    for (MatchSlot* s : batch) in.push_back(s->leaf.input());
    ev->evaluate(in, out);
    for (size_t k = 0; k < batch.size(); ++k) batch[k]->treeToMove().commit(batch[k]->leaf, out[k]);
    batch.clear();
  };
  while (active > 0) {
    for (MatchSlot& s : slots) {
      while (s.active) {
        if (s.collect()) {
          (s.moverIsA() ? batchA : batchB).push_back(&s);
          break;
        }
        // Game over: store it and refill the slot.
        r.games[static_cast<size_t>(s.pair * 2 + (s.aIsBlack ? 0 : 1))] = s.game;
        --active;
        if (startNext(s)) ++active;
      }
    }
    flush(batchA, a.ev);
    flush(batchB, b.ev);
  }
}

// The M4a match driver (DESIGN 5.5.1): T search threads over the G slots (thread t owns
// slots t, t+T, ...), one evaluation thread that owns both evaluators and runs one
// forward per side and step — a request of A is never in a forward of B. Per game the
// slot's collect() is the first-version one, so results are independent of T.
void playMatchThreaded(MatchReport& r, MatchPlayer& a, MatchPlayer& b, int n, float komi, int moveCap,
                       const std::vector<Opening>& openings, uint64_t seed, int G, int T, int maxBatch) {
  const int totalGames = static_cast<int>(r.games.size());
  std::vector<MatchSlot> slots(static_cast<size_t>(G));
  EvalQueue queue(T, maxBatch <= 0 ? G : maxBatch);
  std::mutex startMutex;
  int nextGame = 0;
  auto startNext = [&](MatchSlot& s) {
    std::lock_guard<std::mutex> lk(startMutex);
    if (nextGame >= totalGames) return false;
    startMatchGame(s, nextGame++, a, b, n, komi, moveCap, openings, seed);
    return true;
  };
  std::mutex errorMutex;
  std::exception_ptr error;
  auto fail = [&](std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> lk(errorMutex);
      if (!error) error = std::move(e);
    }
    queue.abort();
  };

  auto searchThread = [&](int t) {
    std::vector<MatchSlot*> mine;
    for (int i = t; i < G; i += T) mine.push_back(&slots[static_cast<size_t>(i)]);
    std::vector<NNOutput> outs(mine.size());
    std::vector<EvalRequest> round;
    std::vector<std::pair<MatchSlot*, NNOutput*>> pending;
    auto abandon = [&] {
      for (auto& [s, o] : pending) s->treeToMove().abort(s->leaf);
      pending.clear();
    };
    try {
      int active = 0;
      for (MatchSlot* s : mine)
        if (startNext(*s)) ++active;
      while (active > 0 && !queue.aborted()) {
        round.clear();
        pending.clear();
        for (size_t k = 0; k < mine.size(); ++k) {
          MatchSlot& s = *mine[k];
          while (s.active) {
            if (s.collect()) {
              round.push_back(EvalRequest{s.leaf.planes.data(), &outs[k], t, s.moverIsA() ? 0 : 1});
              pending.emplace_back(&s, &outs[k]);
              break;
            }
            r.games[static_cast<size_t>(s.pair * 2 + (s.aIsBlack ? 0 : 1))] = s.game;  // distinct index per game
            --active;
            if (startNext(s)) ++active;
          }
        }
        if (round.empty()) continue;
        if (!queue.submitRound(t, round)) {
          abandon();
          return;
        }
        for (auto& [s, o] : pending) s->treeToMove().commit(s->leaf, *o);
        pending.clear();
      }
    } catch (...) {
      fail(std::current_exception());
      abandon();
    }
  };

  EvalThreadStats evalStats;
  auto evaluationThread = [&] {
    std::vector<EvalRequest> batch;
    std::vector<size_t> bucket;
    std::vector<NNInput> in;
    std::vector<NNOutput> out;
    try {
      while (queue.takeBatch(batch)) {
        for (int side = 0; side < 2; ++side) {
          bucket.clear();
          for (size_t i = 0; i < batch.size(); ++i)
            if (batch[i].side == side) bucket.push_back(i);
          if (bucket.empty()) continue;
          NNEvaluator* ev = side == 0 ? a.ev : b.ev;
          if (ev == nullptr) throw std::logic_error("a random player has no pending leaves");
          in.clear();
          for (size_t i : bucket) in.push_back(NNInput{batch[i].planes});
          evaluateWithRetry(*ev, in, out, evalStats);
          for (size_t k = 0; k < bucket.size(); ++k) *batch[bucket[k]].out = std::move(out[k]);
          evalStats.batches += 1;
          evalStats.evaluations += static_cast<uint64_t>(bucket.size());
        }
        queue.deliver(batch);
      }
    } catch (...) {
      fail(std::current_exception());
    }
  };

  std::thread evaluator(evaluationThread);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) workers.emplace_back(searchThread, t);
  for (std::thread& w : workers) w.join();
  queue.stop();
  evaluator.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace

MatchReport playMatch(MatchPlayer a, MatchPlayer b, int n, float komi, int moveCap, const std::vector<Opening>& openings,
                      uint64_t seed, int gamesInFlight, int threads, int maxBatch) {
  MatchReport r;
  r.nameA = a.name;
  r.nameB = b.name;
  r.boardSize = n;
  r.komi = komi;
  r.simulations = a.random ? b.params.simulations : a.params.simulations;
  if (a.random && b.random && !openings.empty()) r.simulations = 0;
  r.seed = seed;
  r.openings = openings;
  const int totalGames = static_cast<int>(openings.size()) * 2;
  r.games.resize(static_cast<size_t>(totalGames));
  const int G = gamesInFlight <= 0 ? std::max(1, totalGames) : std::min(gamesInFlight, std::max(1, totalGames));
  const int T = std::max(1, std::min(threads, G));
  if (T == 1) playMatchSingleThread(r, a, b, n, komi, moveCap, openings, seed, G);
  else playMatchThreaded(r, a, b, n, komi, moveCap, openings, seed, G, T, maxBatch);
  std::set<uint64_t> trajectories;
  for (size_t i = 0; i < openings.size(); ++i) {
    float pairScore = 0.0f;
    for (int side = 0; side < 2; ++side) {
      const MatchGame& g = r.games[i * 2 + static_cast<size_t>(side)];
      const float s = g.scoreForA();
      pairScore += s;
      if (s > 0.75f) ++r.winsA;
      else if (s < 0.25f) ++r.lossesA;
      else ++r.drawsA;
      trajectories.insert(trajectoryHash(g.moves));
    }
    r.pairScores.push_back(pairScore / 2.0f);
  }
  r.uniqueTrajectories = static_cast<int>(trajectories.size());
  double sum = 0.0;
  for (float s : r.pairScores) sum += s;
  r.meanPairScore = r.pairScores.empty() ? 0.0 : sum / static_cast<double>(r.pairScores.size());
  auto ci = bootstrapMeanInterval(r.pairScores, 10000, deriveSeed(seed, 0xB007ull));
  r.ciLow = ci.first;
  r.ciHigh = ci.second;
  return r;
}

std::string matchReportToJson(const MatchReport& r) {
  nlohmann::json j;
  j["a"] = r.nameA;
  j["b"] = r.nameB;
  j["board_size"] = r.boardSize;
  j["komi"] = r.komi;
  j["simulations"] = r.simulations;
  j["seed"] = r.seed;
  j["pairs"] = r.pairScores.size();
  j["games"] = r.games.size();
  j["wins_a"] = r.winsA;
  j["losses_a"] = r.lossesA;
  j["draws_a"] = r.drawsA;
  j["mean_pair_score"] = r.meanPairScore;
  j["ci95"] = {r.ciLow, r.ciHigh};
  j["unique_trajectories"] = r.uniqueTrajectories;
  j["pair_scores"] = r.pairScores;
  nlohmann::json ops = nlohmann::json::array();
  for (const Opening& o : r.openings) ops.push_back(movesToJson(o.moves, r.boardSize));
  j["openings"] = ops;
  nlohmann::json games = nlohmann::json::array();
  for (const MatchGame& g : r.games) {
    games.push_back({{"pair", g.pair},
                     {"a_is_black", g.aIsBlack},
                     {"result", g.result},
                     {"score", g.score},
                     {"termination", static_cast<int>(g.termination)},
                     {"moves", movesToJson(g.moves, r.boardSize)}});
  }
  j["games_detail"] = games;
  return j.dump(2);
}

}  // namespace mango
