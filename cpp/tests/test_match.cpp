// Match driver (DESIGN 5.8): openings, pair scores, colour swap, bootstrap interval,
// report. FakeEvaluator only.
#include <cmath>
#include <set>
#include <vector>

#include "core/board.h"
#include "doctest/doctest.h"
#include "match/match.h"
#include "nlohmann/json.hpp"
#include "nn/evaluator.h"

using namespace mango;

namespace {

SearchParams evalParams(int sims) {
  SearchConfig c;
  c.evalSimulations = sims;
  c.searchSymmetry = false;
  return SearchParams::fromConfig(c, 5, /*selfplay=*/false);
}

bool blackToMove(const uint8_t* planes, int n) { return planes[16 * n * n] == 1; }

// Area score of the position encoded in the planes, from the mover's perspective,
// squashed to (-1, 1): a crude but real signal that rewards captures and territory.
FakeEvaluator scoreAwareEvaluator(int n, float komi) {
  return FakeEvaluator(n, [n, komi](const uint8_t* planes, float* policy, float* value) {
    const int nn = n * n;
    Board b(n, komi);
    std::string diagram;
    const bool black = blackToMove(planes, n);
    for (int p = 0; p < nn; ++p) {
      const bool mover = planes[p] == 1, opp = planes[8 * nn + p] == 1;
      char ch = '.';
      if (mover) ch = black ? 'X' : 'O';
      if (opp) ch = black ? 'O' : 'X';
      diagram += ch;
      diagram += (p % n == n - 1) ? '\n' : ' ';
    }
    b = Board::fromString(diagram, komi, black ? Color::Black : Color::White);
    float s = b.score();
    if (!black) s = -s;
    *value = std::tanh(s / 5.0f);
    for (int a = 0; a < nn; ++a) policy[a] = 1.0f;
    policy[nn] = 0.05f;
  });
}

}  // namespace

TEST_CASE("random openings are distinct, legal and seed-deterministic") {
  auto ops = generateRandomOpenings(5, 7.5f, 50, 12, 2, 99);
  REQUIRE(ops.size() == 12);
  std::set<std::vector<Move>> seen;
  for (const Opening& o : ops) {
    REQUIRE(o.moves.size() == 2);
    CHECK(seen.insert(o.moves).second);
    Board b(5, 7.5f);
    GameHistory h;
    h.reset(b.hash());
    for (Move m : o.moves) {
      CHECK(m != kPass);
      REQUIRE(b.isLegal(m, HashHistory(h)));
      b.play(m);
      h.push(m, b.hash());
    }
  }
  auto again = generateRandomOpenings(5, 7.5f, 50, 12, 2, 99);
  for (size_t i = 0; i < ops.size(); ++i) CHECK(again[i].moves == ops[i].moves);
  auto other = generateRandomOpenings(5, 7.5f, 50, 12, 2, 100);
  bool differs = false;
  for (size_t i = 0; i < ops.size(); ++i) differs |= (other[i].moves != ops[i].moves);
  CHECK(differs);
  // More distinct openings than a 2x2 board can provide: fewer are returned, none repeated.
  auto few = generateRandomOpenings(2, 0.5f, 8, 50, 1, 1);
  CHECK(few.size() == 4);
}

TEST_CASE("bootstrap interval of pair scores") {
  std::vector<float> constant(30, 0.5f);
  auto ci = bootstrapMeanInterval(constant, 1000, 1);
  CHECK(ci.first == doctest::Approx(0.5));
  CHECK(ci.second == doctest::Approx(0.5));
  std::vector<float> mixed;
  for (int i = 0; i < 40; ++i) mixed.push_back(i % 2 ? 1.0f : 0.0f);
  ci = bootstrapMeanInterval(mixed, 10000, 2);
  CHECK(ci.first < 0.5);
  CHECK(ci.second > 0.5);
  CHECK(ci.first > 0.25);
  CHECK(ci.second < 0.75);
  auto same = bootstrapMeanInterval(mixed, 10000, 2);
  CHECK(same == ci);
  std::vector<float> ones(20, 1.0f);
  ci = bootstrapMeanInterval(ones, 100, 3);
  CHECK(ci.first == doctest::Approx(1.0));
  CHECK(bootstrapMeanInterval({}, 10, 1) == std::pair<double, double>(0.0, 0.0));
}

TEST_CASE("match plays every opening twice with colours swapped and scores on pairs") {
  const int n = 5;
  FakeEvaluator evA(n, 0.0f), evB(n, 0.0f);
  MatchPlayer a{&evA, evalParams(8), "A"};
  MatchPlayer b{&evB, evalParams(8), "B"};
  auto openings = generateRandomOpenings(n, 7.5f, 50, 5, 2, 4);
  MatchReport r = playMatch(a, b, n, 7.5f, 50, openings, 11);
  REQUIRE(r.games.size() == 10);
  REQUIRE(r.pairScores.size() == 5);
  double mean = 0.0;
  for (size_t i = 0; i < 5; ++i) {
    const MatchGame& g0 = r.games[2 * i];
    const MatchGame& g1 = r.games[2 * i + 1];
    CHECK(g0.pair == static_cast<int>(i));
    CHECK(g1.pair == static_cast<int>(i));
    CHECK(g0.aIsBlack);
    CHECK_FALSE(g1.aIsBlack);
    for (const MatchGame* g : {&g0, &g1}) {
      REQUIRE(g->moves.size() >= openings[i].moves.size());
      for (size_t k = 0; k < openings[i].moves.size(); ++k) CHECK(g->moves[k] == openings[i].moves[k]);
      // Replay: the game is legal and its result matches the final position.
      Board bd(n, 7.5f, 50);
      GameHistory h;
      h.reset(bd.hash());
      for (Move m : g->moves) {
        REQUIRE(bd.isLegal(m, HashHistory(h)));
        bd.play(m);
        h.push(m, bd.hash());
      }
      CHECK(bd.gameOver());
      CHECK(g->score == doctest::Approx(bd.score()));
      CHECK(g->result == (g->score > 0 ? 1 : (g->score < 0 ? -1 : 0)));
      // scoreForA follows the colour A had in this game.
      const float expected = g->result == 0 ? 0.5f : ((g->result > 0) == g->aIsBlack ? 1.0f : 0.0f);
      CHECK(g->scoreForA() == expected);
    }
    CHECK(r.pairScores[i] == doctest::Approx((g0.scoreForA() + g1.scoreForA()) / 2.0f));
    mean += r.pairScores[i];
  }
  CHECK(r.meanPairScore == doctest::Approx(mean / 5.0));
  CHECK(r.winsA + r.lossesA + r.drawsA == 10);
  CHECK(r.uniqueTrajectories >= 1);
  CHECK(r.uniqueTrajectories <= 10);
  CHECK(r.ciLow <= r.meanPairScore);
  CHECK(r.ciHigh >= r.meanPairScore);
  // Deterministic under the same seed.
  MatchReport r2 = playMatch(a, b, n, 7.5f, 50, openings, 11);
  CHECK(r2.pairScores == r.pairScores);
  for (size_t i = 0; i < r.games.size(); ++i) CHECK(r2.games[i].moves == r.games[i].moves);
  // Report JSON carries the numbers.
  nlohmann::json j = nlohmann::json::parse(matchReportToJson(r));
  CHECK(j["pairs"] == 5);
  CHECK(j["games"] == 10);
  CHECK(j["mean_pair_score"].get<double>() == doctest::Approx(r.meanPairScore));
  CHECK(j["openings"].size() == 5);
  CHECK(j["games_detail"].size() == 10);
}

TEST_CASE("a score-aware player beats a passive one from both colours") {
  const int n = 5;
  FakeEvaluator strong = scoreAwareEvaluator(n, 7.5f);
  // The passive player only ever passes (prior mass on pass, zero value).
  FakeEvaluator passive(n, [n](const uint8_t*, float* policy, float* value) {
    for (int a = 0; a < n * n; ++a) policy[a] = 0.0f;
    policy[n * n] = 1.0f;
    *value = 0.0f;
  });
  MatchPlayer a{&strong, evalParams(24), "strong"};
  MatchPlayer b{&passive, evalParams(24), "passive"};
  auto openings = generateRandomOpenings(n, 7.5f, 50, 6, 2, 8);
  MatchReport r = playMatch(a, b, n, 7.5f, 50, openings, 21);
  CHECK(r.meanPairScore == doctest::Approx(1.0));
  CHECK(r.ciLow == doctest::Approx(1.0));
  CHECK(r.lossesA == 0);
  // And the reverse assignment gives the mirror image.
  MatchReport rev = playMatch(b, a, n, 7.5f, 50, openings, 21);
  CHECK(rev.meanPairScore == doctest::Approx(0.0));
  CHECK(rev.winsA == 0);
}
