#include "match/match.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <sstream>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"
#include "core/zobrist.h"
#include "nlohmann/json.hpp"
#include "search/mcts.h"

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

namespace {

// One game in flight: both sides' trees follow the same board.
struct MatchSlot {
  int pair = 0;
  bool aIsBlack = true;
  Board board;
  GameHistory hist;
  std::unique_ptr<SearchTree> tb, tw;
  MatchGame game;
  PendingLeaf leaf;
  bool active = false;
  MatchSlot() : board(2, 0.0f) {}

  SearchTree& treeToMove() { return board.toMove() == Color::Black ? *tb : *tw; }
  bool moverIsA() const { return (board.toMove() == Color::Black) == aIsBlack; }

  void start(int pairIdx, bool aBlack, MatchPlayer& black, MatchPlayer& white, int n, float komi, int moveCap,
             const Opening& opening, uint64_t seedBlack, uint64_t seedWhite) {
    pair = pairIdx;
    aIsBlack = aBlack;
    board = Board(n, komi, moveCap);
    hist.reset(board.hash());
    tb = std::make_unique<SearchTree>(black.params, n, seedBlack);
    tw = std::make_unique<SearchTree>(white.params, n, seedWhite);
    tb->newGame(board, hist);
    tw->newGame(board, hist);
    game = MatchGame();
    game.pair = pairIdx;
    game.aIsBlack = aBlack;
    for (Move m : opening.moves) apply(m);
    active = true;
    if (!board.gameOver()) treeToMove().prepareRoot();
  }
  void apply(Move m) {
    board.play(m);
    hist.push(m, board.hash());
    tb->advance(m, board, hist);
    tw->advance(m, board, hist);
    game.moves.push_back(m);
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
      SearchTree& tree = treeToMove();
      if (!tree.rootExpanded()) {
        if (tree.collectLeaf(leaf) != CollectResult::Pending) throw std::logic_error("unexpected collect result at the root");
        return true;
      }
      if (tree.budgetExhausted()) {
        apply(tree.selectMove(board.moveCount()));
        if (board.gameOver()) {
          finishGame();
          return false;
        }
        treeToMove().prepareRoot();
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

MatchReport playMatch(MatchPlayer a, MatchPlayer b, int n, float komi, int moveCap, const std::vector<Opening>& openings,
                      uint64_t seed, int gamesInFlight) {
  MatchReport r;
  r.nameA = a.name;
  r.nameB = b.name;
  r.boardSize = n;
  r.komi = komi;
  r.simulations = a.params.simulations;
  r.seed = seed;
  r.openings = openings;
  const int totalGames = static_cast<int>(openings.size()) * 2;
  r.games.resize(static_cast<size_t>(totalGames));
  const int G = gamesInFlight <= 0 ? std::max(1, totalGames) : std::min(gamesInFlight, std::max(1, totalGames));
  std::vector<MatchSlot> slots(static_cast<size_t>(G));
  int nextGame = 0;
  auto startNext = [&](MatchSlot& s) {
    if (nextGame >= totalGames) return false;
    const int gi = nextGame++;
    const size_t i = static_cast<size_t>(gi / 2);
    const int side = gi % 2;
    const bool aIsBlack = side == 0;
    MatchPlayer& black = aIsBlack ? a : b;
    MatchPlayer& white = aIsBlack ? b : a;
    s.start(static_cast<int>(i), aIsBlack, black, white, n, komi, moveCap, openings[i], deriveSeed(seed, i * 4 + side * 2 + 0),
            deriveSeed(seed, i * 4 + side * 2 + 1));
    return true;
  };
  int active = 0;
  for (MatchSlot& s : slots)
    if (startNext(s)) ++active;
  std::vector<MatchSlot*> batchA, batchB;
  std::vector<NNInput> in;
  std::vector<NNOutput> out;
  auto flush = [&](std::vector<MatchSlot*>& batch, NNEvaluator& ev) {
    if (batch.empty()) return;
    in.clear();
    for (MatchSlot* s : batch) in.push_back(s->leaf.input());
    ev.evaluate(in, out);
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
    flush(batchA, *a.ev);
    flush(batchB, *b.ev);
  }
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
  for (const Opening& o : r.openings) ops.push_back(o.moves);
  j["openings"] = ops;
  nlohmann::json games = nlohmann::json::array();
  for (const MatchGame& g : r.games) {
    games.push_back({{"pair", g.pair},
                     {"a_is_black", g.aIsBlack},
                     {"result", g.result},
                     {"score", g.score},
                     {"termination", static_cast<int>(g.termination)},
                     {"moves", g.moves}});
  }
  j["games_detail"] = games;
  return j.dump(2);
}

}  // namespace mango
