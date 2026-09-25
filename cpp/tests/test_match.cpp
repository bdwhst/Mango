// Match driver (DESIGN 5.8): openings, pair scores, colour swap, bootstrap interval,
// report. FakeEvaluator only.
#include <cmath>
#include <set>
#include <stdexcept>
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
  MatchReport r = playMatch(a, b, n, 7.5f, 50, openings, 11, /*gamesInFlight=*/1);
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
  // Deterministic under the same seed, and independent of how many games are in flight
  // (K = 1 per game: every game's search is exactly the sequential search).
  for (int inFlight : {1, 3, 10}) {
    CAPTURE(inFlight);
    MatchReport r2 = playMatch(a, b, n, 7.5f, 50, openings, 11, inFlight);
    CHECK(r2.pairScores == r.pairScores);
    REQUIRE(r2.games.size() == r.games.size());
    for (size_t i = 0; i < r.games.size(); ++i) {
      CHECK(r2.games[i].moves == r.games[i].moves);
      CHECK(r2.games[i].pair == r.games[i].pair);
      CHECK(r2.games[i].aIsBlack == r.games[i].aIsBlack);
    }
  }
  // Report JSON carries the numbers.
  nlohmann::json j = nlohmann::json::parse(matchReportToJson(r));
  CHECK(j["pairs"] == 5);
  CHECK(j["games"] == 10);
  CHECK(j["mean_pair_score"].get<double>() == doctest::Approx(r.meanPairScore));
  CHECK(j["openings"].size() == 5);
  CHECK(j["games_detail"].size() == 10);
}

TEST_CASE("openings survive a JSON round trip and illegal ones are rejected") {
  auto ops = generateRandomOpenings(5, 7.5f, 50, 6, 3, 17);
  ops.push_back(Opening{{kPass, static_cast<Move>(12)}});  // pass is written as n*n, never -1
  const std::string text = openingsToJson(ops, 5);
  CHECK(text.find("-1") == std::string::npos);
  CHECK(text.find("[25,12]") != std::string::npos);
  auto back = openingsFromJson(text, 5, 7.5f, 50);
  REQUIRE(back.size() == ops.size());
  for (size_t i = 0; i < ops.size(); ++i) CHECK(back[i].moves == ops[i].moves);
  // The report uses the same encoding, so its openings and moves can be fed back in.
  FakeEvaluator ev(5, 0.0f);
  MatchPlayer p{&ev, evalParams(4), "p"};
  MatchReport r = playMatch(p, p, 5, 7.5f, 50, {ops.back()}, 3, 1);
  nlohmann::json j = nlohmann::json::parse(matchReportToJson(r));
  CHECK(j["openings"][0] == nlohmann::json({25, 12}));
  for (const auto& g : j["games_detail"]) {
    for (int m : g["moves"].get<std::vector<int>>()) {
      CHECK(m >= 0);
      CHECK(m <= 25);
    }
    CHECK(g["moves"][0] == 25);
    // The whole game replays as an "opening" (two passes end it, legal throughout).
    CHECK_NOTHROW(openingsFromJson(nlohmann::json::array({g["moves"]}).dump(), 5, 7.5f, 50));
  }
  CHECK_NOTHROW(openingsFromJson(j["openings"].dump(), 5, 7.5f, 50));
  CHECK_THROWS_AS(openingsFromJson("[[0, 0]]", 5, 7.5f, 50), std::invalid_argument);     // occupied point
  CHECK_THROWS_AS(openingsFromJson("[[26]]", 5, 7.5f, 50), std::invalid_argument);       // out of range
  CHECK_THROWS_AS(openingsFromJson("[[0], 3]", 5, 7.5f, 50), std::invalid_argument);     // not an array
  CHECK_THROWS_AS(openingsFromJson("nope", 5, 7.5f, 50), std::invalid_argument);         // not JSON
  CHECK(openingsFromJson("[[25, 25]]", 5, 7.5f, 50)[0].moves == std::vector<Move>{kPass, kPass});  // legal (game over)
}

TEST_CASE("the random anchor plays legal games, is seed-deterministic and loses to a searcher") {
  const int n = 5;
  const SearchParams params = evalParams(16);
  MatchPlayer ra{nullptr, params, "random", true};
  MatchPlayer rb{nullptr, params, "random", true};
  auto openings = generateRandomOpenings(n, 7.5f, 50, 4, 2, 5);
  MatchReport r = playMatch(ra, rb, n, 7.5f, 50, openings, 3, 2);
  REQUIRE(r.games.size() == 8);
  for (const MatchGame& g : r.games) {
    Board bd(n, 7.5f, 50);
    GameHistory h;
    h.reset(bd.hash());
    for (size_t k = 0; k < g.moves.size(); ++k) {
      const Move m = g.moves[k];
      REQUIRE(bd.isLegal(m, HashHistory(h)));
      if (m == kPass && k >= openings[static_cast<size_t>(g.pair)].moves.size()) {
        // A random side passes only when no board move is legal.
        std::vector<Move> legal;
        bd.legalMoves(HashHistory(h), legal);
        CHECK(legal.size() == 1);
      }
      bd.play(m);
      h.push(m, bd.hash());
    }
    CHECK(bd.gameOver());
    CHECK(g.score == doctest::Approx(bd.score()));
  }
  MatchReport again = playMatch(ra, rb, n, 7.5f, 50, openings, 3, 1);
  for (size_t i = 0; i < r.games.size(); ++i) CHECK(again.games[i].moves == r.games[i].moves);
  MatchReport other = playMatch(ra, rb, n, 7.5f, 50, openings, 4, 1);
  bool differs = false;
  for (size_t i = 0; i < r.games.size(); ++i) differs |= (other.games[i].moves != r.games[i].moves);
  CHECK(differs);
  // A (crude) score-aware searcher clearly beats the anchor from both colours; the
  // crude evaluator with 24 simulations is not perfect on 5x5, so the bar is 0.7.
  FakeEvaluator strong = scoreAwareEvaluator(n, 7.5f);
  MatchPlayer s{&strong, evalParams(24), "strong"};
  MatchReport sv = playMatch(s, rb, n, 7.5f, 50, generateRandomOpenings(n, 7.5f, 50, 6, 2, 8), 21, 3);
  CHECK(sv.meanPairScore > 0.7);
  CHECK(sv.nameB == "random");
  MatchReport vs = playMatch(ra, s, n, 7.5f, 50, generateRandomOpenings(n, 7.5f, 50, 6, 2, 8), 21, 3);
  CHECK(vs.meanPairScore < 0.3);
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

// ---------------------------------------------------------------------------------
// Multi-threaded match driver (DESIGN 5.5.1, M4a): results identical for every thread
// count and batch cap, the two sides never share a forward, every forward respects the cap.
#include "tests/test_util.h"

TEST_CASE("threaded match equals the single-threaded match and keeps the sides apart") {
  const int n = 5;
  // Side-specific evaluators: mixing a request of A into a forward of B (or vice versa)
  // would change the result.
  FakeEvaluator innerA(n, [n](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f + static_cast<float>(a % 3);
    *value = blackToMove(planes, n) ? 0.3f : -0.3f;
  });
  FakeEvaluator innerB = scoreAwareEvaluator(n, 7.5f);
  test::RecordingEvaluator evA(innerA), evB(innerB);
  MatchPlayer a{&evA, evalParams(8), "A"};
  MatchPlayer b{&evB, evalParams(8), "B"};
  auto openings = generateRandomOpenings(n, 7.5f, 50, 6, 2, 21);
  MatchReport ref = playMatch(a, b, n, 7.5f, 50, openings, 5, /*gamesInFlight=*/4, /*threads=*/1);
  CHECK(evA.calls() > 0);
  CHECK(evB.calls() > 0);
  struct Cfg {
    int G, threads, maxBatch;
  };
  for (const Cfg c : {Cfg{4, 4, 0}, Cfg{12, 3, 5}, Cfg{6, 2, 1}, Cfg{5, 8, 0}}) {
    CAPTURE(c.G);
    CAPTURE(c.threads);
    CAPTURE(c.maxBatch);
    const int beforeA = evA.calls(), beforeB = evB.calls();
    MatchReport r = playMatch(a, b, n, 7.5f, 50, openings, 5, c.G, c.threads, c.maxBatch);
    CHECK(r.pairScores == ref.pairScores);
    CHECK(r.meanPairScore == ref.meanPairScore);
    REQUIRE(r.games.size() == ref.games.size());
    for (size_t i = 0; i < r.games.size(); ++i) {
      CHECK(r.games[i].moves == ref.games[i].moves);
      CHECK(r.games[i].result == ref.games[i].result);
      CHECK(r.games[i].pair == ref.games[i].pair);
      CHECK(r.games[i].aIsBlack == ref.games[i].aIsBlack);
    }
    CHECK(evA.calls() > beforeA);
    CHECK(evB.calls() > beforeB);
    const int cap = c.maxBatch > 0 ? c.maxBatch : std::min(c.G, static_cast<int>(openings.size()) * 2);
    for (size_t k = static_cast<size_t>(beforeA); k < evA.sizes().size(); ++k) CHECK(evA.sizes()[k] <= cap);
    for (size_t k = static_cast<size_t>(beforeB); k < evB.sizes().size(); ++k) CHECK(evB.sizes()[k] <= cap);
  }
  // A random side has no requests; the threaded driver handles it like the first version.
  MatchPlayer rnd{nullptr, evalParams(8), "random", true};
  MatchReport r1 = playMatch(a, rnd, n, 7.5f, 50, openings, 9, 4, 1);
  MatchReport r4 = playMatch(a, rnd, n, 7.5f, 50, openings, 9, 4, 4, 2);
  CHECK(r1.pairScores == r4.pairScores);
  for (size_t i = 0; i < r1.games.size(); ++i) CHECK(r1.games[i].moves == r4.games[i].moves);
  MatchReport rr = playMatch(rnd, rnd, n, 7.5f, 50, openings, 9, 4, 3);
  CHECK(rr.games.size() == openings.size() * 2);
  // One failure of a side's forward is retried (same result); two in a row abort the match.
  evB.failAt = {evB.calls() + 2};
  MatchReport retried = playMatch(a, b, n, 7.5f, 50, openings, 5, 6, 3, 0);
  CHECK(retried.pairScores == ref.pairScores);
  evB.failAt = {evB.calls() + 2, evB.calls() + 3};
  CHECK_THROWS_AS(playMatch(a, b, n, 7.5f, 50, openings, 5, 6, 3, 0), std::runtime_error);
  evB.failAt.clear();
}
