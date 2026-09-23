// Sequential self-play game runner (DESIGN 5.5): record consistency, determinism,
// resignation, move cap, no-resign tagging. FakeEvaluator only.
#include <cmath>
#include <vector>

#include "core/board.h"
#include "doctest/doctest.h"
#include "nn/evaluator.h"
#include "selfplay/chunk.h"
#include "selfplay/game_runner.h"

using namespace mango;

namespace {

SearchParams selfplayParams(int sims) {
  SearchConfig c;
  c.simulations = sims;
  c.temperatureMoves = 4;
  c.searchSymmetry = false;
  return SearchParams::fromConfig(c, 5, /*selfplay=*/true);
}

bool blackToMove(const uint8_t* planes, int n) { return planes[16 * n * n] == 1; }

// Replays the record's moves from the empty board and checks every stored field
// against the replay.
void checkRecordConsistent(const GameRecord& g, int n, float komi, int moveCap, const ChunkHeader& h) {
  REQUIRE(validateGame(g, h) == "");
  const int nn = n * n;
  Board b(n, komi, moveCap);
  std::vector<Color> cells(nn);
  auto packed = [&](std::vector<uint8_t>& out) {
    for (int p = 0; p < nn; ++p) cells[p] = b.atPoint(p);
    out.resize(packedSnapshotBytes(n));
    packSnapshot(cells.data(), n, out.data());
  };
  std::vector<uint8_t> snap;
  for (int t = 0; t < g.T(); ++t) {
    packed(snap);
    CHECK(snap == g.snapshots[t]);
    const uint16_t m = g.moves[t];
    const Move mv = m == nn ? kPass : static_cast<Move>(m);
    REQUIRE(b.isLegal(mv, HashHistory()));
    // The played move is among the visited actions, and counts sum to the total.
    bool played = false;
    uint64_t sum = 0;
    for (const auto& [a, c] : g.visits[t].counts) {
      played |= (a == m);
      sum += c;
    }
    CHECK(played);
    CHECK(sum == g.visits[t].rootTotalVisits);
    CHECK(g.rootValue[t] >= -1.0f);
    CHECK(g.rootValue[t] <= 1.0f);
    b.play(mv);
  }
  packed(snap);
  CHECK(snap == g.snapshots.back());
  CHECK(g.score == doctest::Approx(b.score()));
  if (g.termination != Termination::Resign) {
    CHECK(g.result == (g.score > 0 ? 1 : (g.score < 0 ? -1 : 0)));
    CHECK(g.termination == (b.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap));
    CHECK(b.gameOver());
  }
}

}  // namespace

TEST_CASE("self-play record is consistent with a replay of its moves") {
  const int n = 5;
  FakeEvaluator ev(n, 0.0f);
  SelfplayGameOptions opt;
  opt.boardSize = n;
  opt.komi = 7.5f;
  opt.moveCap = 50;
  opt.params = selfplayParams(20);
  opt.gameSeed = 123;
  opt.storeFinalOwnership = true;
  SelfplayGameResult r = playSelfplayGame(ev, opt, "0000-test");
  ChunkHeader h;
  h.boardSize = n;
  h.recordExtras = kExtrasFinalOwnership;
  checkRecordConsistent(r.record, n, 7.5f, 50, h);
  CHECK(r.record.T() > 0);
  CHECK(r.record.gameSeed == 123);
  CHECK_FALSE(r.record.noResignGame);
  // Every move ran exactly 20 new simulations; with tree reuse the total can be larger.
  for (const MoveVisits& v : r.record.visits) CHECK(v.rootTotalVisits >= 20);
  // Ownership matches the final position.
  Board b(n, 7.5f, 50);
  for (uint16_t m : r.record.moves) b.play(m == 25 ? kPass : static_cast<Move>(m));
  std::vector<int8_t> own(25);
  b.areaOwnership(own.data());
  CHECK(r.record.finalOwnership == own);
  // SGF mirrors the record.
  CHECK(r.sgf.size == n);
  CHECK(r.sgf.moves.size() == r.record.moves.size());
  CHECK(r.sgf.comment.find("game_seed=123") != std::string::npos);
  CHECK(r.evaluations > 0);
  CHECK(r.evaluations <= r.record.T() * 21);
}

TEST_CASE("self-play games are reproducible from their seed") {
  const int n = 5;
  FakeEvaluator ev(n, 0.0f);
  SelfplayGameOptions opt;
  opt.boardSize = n;
  opt.moveCap = 50;
  opt.params = selfplayParams(16);
  opt.gameSeed = 9;
  SelfplayGameResult a = playSelfplayGame(ev, opt, "m");
  SelfplayGameResult b = playSelfplayGame(ev, opt, "m");
  CHECK(a.record.moves == b.record.moves);
  CHECK(a.record.snapshots == b.record.snapshots);
  CHECK(a.record.rootValue == b.record.rootValue);
  opt.gameSeed = 10;
  SelfplayGameResult c = playSelfplayGame(ev, opt, "m");
  CHECK(a.record.moves != c.record.moves);  // noise and temperature sampling differ
}

TEST_CASE("self-play resignation ends the game for the loser unless it is a no-resign game") {
  const int n = 5;
  // Black is losing everywhere (value from the mover's perspective).
  FakeEvaluator ev(n, [n](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    *value = blackToMove(planes, n) ? -0.95f : 0.95f;
  });
  SelfplayGameOptions opt;
  opt.boardSize = n;
  opt.moveCap = 50;
  opt.params = selfplayParams(10);
  opt.params.resignThreshold = -0.9f;
  opt.gameSeed = 5;
  SelfplayGameResult r = playSelfplayGame(ev, opt, "m");
  CHECK(r.record.termination == Termination::Resign);
  CHECK(r.record.T() == 0);  // black resigns before its first move
  CHECK(r.record.result == -1);
  CHECK(r.record.snapshots.size() == 1);
  CHECK(r.sgf.result == "W+R");
  ChunkHeader h;
  h.boardSize = n;
  CHECK(validateGame(r.record, h) == "");

  opt.noResignGame = true;
  SelfplayGameResult played = playSelfplayGame(ev, opt, "m");
  CHECK(played.record.noResignGame);
  CHECK(played.record.termination != Termination::Resign);
  CHECK(played.record.T() > 0);
  checkRecordConsistent(played.record, n, 7.5f, 50, h);
}

TEST_CASE("self-play stops at the move cap") {
  const int n = 5;
  FakeEvaluator ev(n, [n](const uint8_t*, float* policy, float* value) {
    for (int a = 0; a < n * n; ++a) policy[a] = 1.0f;
    policy[n * n] = 0.0f;  // never pass
    *value = 0.0f;
  });
  SelfplayGameOptions opt;
  opt.boardSize = n;
  opt.moveCap = 6;
  opt.params = selfplayParams(8);
  opt.gameSeed = 3;
  SelfplayGameResult r = playSelfplayGame(ev, opt, "m");
  CHECK(r.record.T() == 6);
  CHECK(r.record.termination == Termination::MoveCap);
  ChunkHeader h;
  h.boardSize = n;
  checkRecordConsistent(r.record, n, 7.5f, 6, h);
}

TEST_CASE("no-resign tagging follows the configured fraction and is seed-deterministic") {
  CHECK_FALSE(isNoResignGame(1, 0.0f));
  CHECK(isNoResignGame(1, 1.0f));
  int tagged = 0;
  const int N = 4000;
  for (uint64_t s = 0; s < N; ++s) tagged += isNoResignGame(s, 0.10f);
  CHECK(tagged > N * 0.07);
  CHECK(tagged < N * 0.13);
  CHECK(isNoResignGame(77, 0.10f) == isNoResignGame(77, 0.10f));
}
