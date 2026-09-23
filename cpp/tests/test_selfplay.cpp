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

// ---------------------------------------------------------------------------------
// Batched driver (DESIGN 5.4.5, K = 1 across games): every game equals its sequential run.
#include <stdexcept>

#include "selfplay/batch_runner.h"

namespace {

// Value from the perspective of the player to move, as a function of the position.
FakeEvaluator colorValueEvaluator(int n, float valueForBlack) {
  return FakeEvaluator(n, [n, valueForBlack](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    *value = blackToMove(planes, n) ? valueForBlack : -valueForBlack;
  });
}

bool sameRecord(const GameRecord& a, const GameRecord& b) {
  if (a.moves != b.moves || a.snapshots != b.snapshots || a.rootValue != b.rootValue || a.rootMaxQ != b.rootMaxQ) return false;
  if (a.result != b.result || a.termination != b.termination || a.score != b.score || a.gameSeed != b.gameSeed) return false;
  if (a.visits.size() != b.visits.size()) return false;
  for (size_t t = 0; t < a.visits.size(); ++t)
    if (a.visits[t].rootTotalVisits != b.visits[t].rootTotalVisits || a.visits[t].counts != b.visits[t].counts) return false;
  return true;
}

SelfplayGameOptions optionsFor(int n, int sims, uint64_t seed, bool symmetry) {
  SelfplayGameOptions opt;
  opt.boardSize = n;
  opt.komi = 7.5f;
  opt.moveCap = 2 * n * n;
  opt.params = selfplayParams(sims);
  opt.params.searchSymmetry = symmetry;
  opt.gameSeed = seed;
  return opt;
}

}  // namespace

TEST_CASE("batched self-play reproduces every game's sequential run (K = 1 across games)") {
  const int n = 5;
  for (bool symmetry : {false, true}) {
    CAPTURE(symmetry);
    FakeEvaluator ev = colorValueEvaluator(n, 0.1f);
    const int total = 7;  // more games than slots: slots are refilled
    BatchedSelfplay driver(ev, /*gamesInFlight=*/3, "m");
    std::vector<GameRecord> batched(total);
    std::vector<int> evals(total, 0);
    BatchStats st = driver.run(
        total, [&](int i) { return optionsFor(n, 12, 100 + i, symmetry); },
        [&](int i, SelfplayGameResult&& r) {
          batched[i] = std::move(r.record);
          evals[i] = r.evaluations;
        });
    CHECK(st.games == total);
    CHECK(st.batches > 0);
    CHECK(st.avgBatch() > 1.0);
    CHECK(st.avgBatch() <= 3.0);
    CHECK(st.retries == 0);
    uint64_t positions = 0, evaluations = 0;
    for (int i = 0; i < total; ++i) {
      FakeEvaluator seq = colorValueEvaluator(n, 0.1f);
      SelfplayGameResult s = playSelfplayGame(seq, optionsFor(n, 12, 100 + i, symmetry), "m");
      CHECK_MESSAGE(sameRecord(batched[i], s.record), "game " << i << " differs from its sequential run");
      CHECK(evals[i] == s.evaluations);
      positions += static_cast<uint64_t>(s.record.T());
      evaluations += static_cast<uint64_t>(s.evaluations);
    }
    CHECK(st.positions == positions);
    CHECK(st.evaluations == evaluations);
  }
}

TEST_CASE("batched self-play retries a failed batch once and keeps completed simulations") {
  const int n = 5;
  int failuresLeft = 1;
  int calls = 0;
  FakeEvaluator flaky(n, [&](const uint8_t*, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    *value = 0.0f;
  });
  // Wrap: throw on the 5th evaluator call once.
  class Wrapper : public NNEvaluator {
   public:
    Wrapper(NNEvaluator& inner, int& calls, int& failuresLeft) : inner_(inner), calls_(calls), failures_(failuresLeft) {}
    int boardSize() const override { return inner_.boardSize(); }
    const std::string& modelId() const override { return inner_.modelId(); }
    void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override {
      ++calls_;
      if (calls_ == 5 && failures_ > 0) {
        --failures_;
        throw std::runtime_error("simulated failure");
      }
      inner_.evaluate(in, out);
    }

   private:
    NNEvaluator& inner_;
    int& calls_;
    int& failures_;
  } ev(flaky, calls, failuresLeft);
  // With search symmetry ON: a retry must reuse the original requests, otherwise the
  // re-collected leaves draw new symmetries and the games diverge from their seeds.
  for (bool symmetry : {true, false}) {
    CAPTURE(symmetry);
    calls = 0;
    failuresLeft = 1;
    BatchedSelfplay driver(ev, 2, "m");
    std::vector<GameRecord> got(2);
    BatchStats st = driver.run(
        2, [&](int i) { return optionsFor(n, 10, 300 + i, symmetry); },
        [&](int i, SelfplayGameResult&& r) { got[i] = std::move(r.record); });
    CHECK(st.retries == 1);
    CHECK(st.games == 2);
    for (int i = 0; i < 2; ++i) {
      FakeEvaluator seq(n, 0.0f);
      SelfplayGameResult s = playSelfplayGame(seq, optionsFor(n, 10, 300 + i, symmetry), "m");
      CHECK_MESSAGE(sameRecord(got[i], s.record), "game " << i << " diverged after the retry");
    }
  }
  // Two consecutive failures abort the run and leave no pending node.
  int always = 1 << 20;
  int calls2 = 0;
  class AlwaysFail : public NNEvaluator {
   public:
    AlwaysFail(int& calls, int& fails) : calls_(calls), fails_(fails) {}
    int boardSize() const override { return 5; }
    const std::string& modelId() const override { return id_; }
    void evaluate(const std::vector<NNInput>&, std::vector<NNOutput>&) override {
      ++calls_;
      if (calls_ >= 3) {
        --fails_;
        throw std::runtime_error("down");
      }
      throw std::runtime_error("down");
    }

   private:
    int& calls_;
    int& fails_;
    std::string id_ = "fail";
  } bad(calls2, always);
  BatchedSelfplay d2(bad, 2, "m");
  CHECK_THROWS(d2.run(2, [&](int i) { return optionsFor(n, 10, 400 + i, false); }, [&](int, SelfplayGameResult&&) {}));
}

TEST_CASE("batched self-play handles games that are over before their first move") {
  const int n = 5;
  FakeEvaluator ev(n, 0.0f);
  BatchedSelfplay driver(ev, 3, "m");
  std::vector<GameRecord> got(5);
  int done = 0;
  BatchStats st = driver.run(
      5,
      [&](int i) {
        SelfplayGameOptions o = optionsFor(n, 8, 500 + i, false);
        o.moveCap = (i % 2 == 0) ? 0 : 4;  // every other game has nothing to play
        return o;
      },
      [&](int i, SelfplayGameResult&& r) {
        got[i] = std::move(r.record);
        ++done;
      });
  CHECK(done == 5);
  CHECK(st.games == 5);
  for (int i = 0; i < 5; ++i) {
    FakeEvaluator seq(n, 0.0f);
    SelfplayGameOptions o = optionsFor(n, 8, 500 + i, false);
    o.moveCap = (i % 2 == 0) ? 0 : 4;
    SelfplayGameResult s = playSelfplayGame(seq, o, "m");
    CHECK(got[i].T() == ((i % 2 == 0) ? 0 : 4));
    CHECK(got[i].termination == Termination::MoveCap);
    CHECK(sameRecord(got[i], s.record));
  }
  CHECK(ev.positions() == static_cast<int>(st.evaluations));
}
