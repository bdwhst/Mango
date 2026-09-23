#include "match/match.h"

#include <algorithm>
#include <set>
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

MatchGame playOne(MatchPlayer& black, MatchPlayer& white, int n, float komi, int moveCap, const Opening& opening,
                  uint64_t seedBlack, uint64_t seedWhite) {
  Board board(n, komi, moveCap);
  GameHistory hist;
  hist.reset(board.hash());
  SearchTree tb(black.params, n, seedBlack);
  SearchTree tw(white.params, n, seedWhite);
  tb.newGame(board, hist);
  tw.newGame(board, hist);
  MatchGame g;
  auto apply = [&](Move m) {
    board.play(m);
    hist.push(m, board.hash());
    tb.advance(m, board, hist);
    tw.advance(m, board, hist);
    g.moves.push_back(m);
  };
  for (Move m : opening.moves) apply(m);
  while (!board.gameOver()) {
    const bool blackToMove = board.toMove() == Color::Black;
    SearchTree& tree = blackToMove ? tb : tw;
    NNEvaluator& ev = *(blackToMove ? black.ev : white.ev);
    tree.runSequential(ev);
    apply(tree.selectMove(board.moveCount()));
  }
  g.score = board.score();
  g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
  g.termination = board.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
  return g;
}

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
                      uint64_t seed) {
  MatchReport r;
  r.nameA = a.name;
  r.nameB = b.name;
  r.boardSize = n;
  r.komi = komi;
  r.simulations = a.params.simulations;
  r.seed = seed;
  r.openings = openings;
  std::set<uint64_t> trajectories;
  for (size_t i = 0; i < openings.size(); ++i) {
    float pairScore = 0.0f;
    for (int side = 0; side < 2; ++side) {
      const bool aIsBlack = side == 0;
      MatchPlayer& black = aIsBlack ? a : b;
      MatchPlayer& white = aIsBlack ? b : a;
      const uint64_t sb = deriveSeed(seed, i * 4 + side * 2 + 0);
      const uint64_t sw = deriveSeed(seed, i * 4 + side * 2 + 1);
      MatchGame g = playOne(black, white, n, komi, moveCap, openings[i], sb, sw);
      g.pair = static_cast<int>(i);
      g.aIsBlack = aIsBlack;
      const float s = g.scoreForA();
      pairScore += s;
      if (s > 0.75f) ++r.winsA;
      else if (s < 0.25f) ++r.lossesA;
      else ++r.drawsA;
      trajectories.insert(trajectoryHash(g.moves));
      r.games.push_back(std::move(g));
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
