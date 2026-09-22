#include <algorithm>
#include <stdexcept>
#include <string>

#include "core/board.h"
#include "core/history.h"
#include "doctest/doctest.h"
#include "tests/test_util.h"

using namespace mango;
using namespace mango::test;

namespace {
Move pt(int r, int c, int n) { return static_cast<Move>(pointOf(r, c, n)); }

RefBoard referenceFor(const Board& b) {
  RefBoard ref(b.size(), b.komi(), b.moveCap());
  for (int p = 0; p < b.numPoints(); ++p) ref.cells[p] = b.atPoint(p);
  ref.toMove = b.toMove();
  ref.positions = {ref.cells};
  return ref;
}

void checkPosition(const Board& b, const RefBoard& ref) {
  CHECK(b.hash() == ref.hash());
  for (int p = 0; p < b.numPoints(); ++p) {
    CAPTURE(p);
    CHECK(b.atPoint(p) == ref.cells[p]);
    CHECK(b.liberties(p) == ref.liberties(p));
    CHECK(b.chainSize(p) == ref.chainSize(p));
  }
}
}  // namespace

TEST_CASE("empty board basics") {
  for (int n : {2, 5, 9, 13, 19}) {
    Board b(n, 7.5f);
    CHECK(b.size() == n);
    CHECK(b.toMove() == Color::Black);
    CHECK(b.hash() == 0);
    CHECK(b.moveCount() == 0);
    CHECK(b.moveCap() == 2 * n * n);
    CHECK_FALSE(b.gameOver());
    GameHistory h;
    h.reset(b.hash());
    std::vector<Move> legal;
    b.legalMoves(HashHistory(h), legal);
    CHECK(static_cast<int>(legal.size()) == n * n + 1);
    CHECK(legal.back() == kPass);
    for (int p = 0; p < n * n; ++p) CHECK(b.liberties(p) == 0);
  }
}

TEST_CASE("capture deduplicates an enemy chain touching the move twice") {
  Board b = Board::fromString("O O X\nO . X\nX X .\n", 0.0f);
  RefBoard ref = referenceFor(b);
  REQUIRE(b.chainSize(0) == 3);
  REQUIRE(b.liberties(0) == 1);
  const Move capture = pt(1, 1, 3);
  REQUIRE(b.isLegal(capture, HashHistory{}));
  uint64_t predicted;
  REQUIRE(b.tryMoveHash(capture, &predicted));
  b.play(capture);
  ref.play(capture);
  CHECK(b.hash() == predicted);
  CHECK(b.get(0, 0) == Color::Empty);
  CHECK(b.get(0, 1) == Color::Empty);
  CHECK(b.get(1, 0) == Color::Empty);
  CHECK(b.chainSize(4) == 5);
  CHECK(b.liberties(4) == 3);
  checkPosition(b, ref);
}

TEST_CASE("capture of distinct chains merges friends and allows reuse of freed points") {
  Board b = Board::fromString(
      ". . X . .\n"
      ". X O X .\n"
      ". X . X .\n"
      ". X O X .\n"
      ". . X . .\n", 0.0f);
  RefBoard ref = referenceFor(b);
  REQUIRE(b.liberties(pointOf(1, 2, 5)) == 1);
  REQUIRE(b.liberties(pointOf(3, 2, 5)) == 1);
  const Move capture = pt(2, 2, 5);
  REQUIRE(b.isLegal(capture, HashHistory{}));
  uint64_t predicted;
  REQUIRE(b.tryMoveHash(capture, &predicted));
  b.play(capture);
  ref.play(capture);
  CHECK(b.hash() == predicted);
  CHECK(b.get(1, 2) == Color::Empty);
  CHECK(b.get(3, 2) == Color::Empty);
  CHECK(b.chainSize(pointOf(2, 2, 5)) == 7);
  checkPosition(b, ref);

  // White passes, then black extends into a captured point and joins the top stone.
  for (Move m : {kPass, pt(1, 2, 5)}) {
    REQUIRE(b.isLegal(m, HashHistory{}));
    REQUIRE(ref.isLegal(m));
    b.play(m);
    ref.play(m);
    checkPosition(b, ref);
  }
  CHECK(b.chainSize(pointOf(1, 2, 5)) == 9);
}

TEST_CASE("filling a friendly chain's last liberty requires a capture") {
  Board suicide = Board::fromString("X X O\nO . O\n. O .\n", 0.0f);
  REQUIRE(suicide.chainSize(0) == 2);
  REQUIRE(suicide.liberties(0) == 1);
  const Board before = suicide;
  uint64_t predicted;
  CHECK_FALSE(suicide.tryMoveHash(pt(1, 1, 3), &predicted));
  CHECK_FALSE(suicide.isLegal(pt(1, 1, 3), HashHistory{}));
  CHECK(suicide == before);

  Board capture = Board::fromString("X X O\nO . O\n. O X\n", 0.0f);
  RefBoard ref = referenceFor(capture);
  REQUIRE(capture.liberties(0) == 1);
  REQUIRE(capture.isLegal(pt(1, 1, 3), HashHistory{}));
  REQUIRE(capture.tryMoveHash(pt(1, 1, 3), &predicted));
  capture.play(pt(1, 1, 3));
  ref.play(pt(1, 1, 3));
  CHECK(capture.hash() == predicted);
  CHECK(capture.chainSize(0) == 3);
  CHECK(capture.liberties(0) == 2);
  CHECK(capture.get(0, 2) == Color::Empty);
  CHECK(capture.get(1, 2) == Color::Empty);
  checkPosition(capture, ref);
}

TEST_CASE("search path superko rejects repetitions but always permits pass") {
  Board b = Board::fromString(
      ". X O . .\nX O . O .\n. X O . .\n. . . . .\n. . . . .\n", 0.0f);
  const uint64_t initial = b.hash();
  REQUIRE(b.isLegal(pt(1, 2, 5), HashHistory{}));
  b.play(pt(1, 2, 5));
  GameHistory game;
  game.reset(b.hash());  // The earlier position exists only in the search path.
  const uint64_t path[] = {b.hash(), initial, b.hash()};
  const Move recapture = pt(1, 1, 5);
  uint64_t predicted;
  REQUIRE(b.tryMoveHash(recapture, &predicted));
  CHECK(predicted == initial);
  CHECK_FALSE(b.isLegal(recapture, HashHistory(game, path, 3)));
  CHECK(b.isLegal(recapture, HashHistory(game, path, 1)));
  CHECK(b.isLegal(recapture, HashHistory(game)));
  Board copy = b;
  copy.play(recapture);  // Locally legal even though the longer history forbids it.
  CHECK(copy.hash() == predicted);
  CHECK(b.isLegal(kPass, HashHistory(game, path, 3)));
  REQUIRE(b.tryMoveHash(kPass, &predicted));
  CHECK(predicted == b.hash());
}

TEST_CASE("corner and edge liberties include only board points") {
  for (int n : {2, 5, 19}) {
    for (int p : {0, n - 1, n * (n - 1), n * n - 1}) {
      Board b(n, 0.0f);
      b.play(static_cast<Move>(p));
      CHECK(b.liberties(p) == 2);
    }
    if (n > 2) {
      for (int p : {1, n, 2 * n - 1, n * (n - 1) + 1}) {
        Board b(n, 0.0f);
        b.play(static_cast<Move>(p));
        CHECK(b.liberties(p) == 3);
      }
    }
  }
  Board b = Board::fromString("O X\n. .\n", 0.0f);
  b.play(pt(1, 0, 2));
  CHECK(b.get(0, 0) == Color::Empty);
  CHECK(b.liberties(1) == 2);
  CHECK(b.liberties(2) == 2);
}

TEST_CASE("legal move lists replace prior contents and contain exactly one final pass") {
  Board b = Board::fromString("X X O\nO . O\n. O .\n", 0.0f);
  GameHistory game;
  game.reset(b.hash());
  CHECK_FALSE(b.isLegal(kNoMove, HashHistory(game)));
  CHECK_FALSE(b.isLegal(static_cast<Move>(b.numPoints()), HashHistory(game)));
  CHECK_FALSE(b.isLegal(0, HashHistory(game)));
  uint64_t predicted = 0;
  CHECK_FALSE(b.tryMoveHash(0, &predicted));
  std::vector<Move> legal{0, 0, kPass, kPass, kNoMove};
  b.legalMoves(HashHistory(game), legal);
  const std::vector<Move> passOnly{kPass};
  CHECK(legal == passOnly);  // Every empty point is suicide.

  Board empty(2, 0.0f);
  REQUIRE(empty.tryMoveHash(1, &predicted));
  const uint64_t path[] = {predicted};
  empty.legalMoves(HashHistory(game, path, 1), legal);
  const std::vector<Move> expected{0, 2, 3, kPass};
  CHECK(legal == expected);
  empty.legalMoves(HashHistory{}, legal);
  const std::vector<Move> all{0, 1, 2, 3, kPass};
  CHECK(legal == all);
}

TEST_CASE("move cap boundaries and pass metadata") {
  Board zero(2, 0.0f, 0);
  CHECK(zero.gameOver());
  CHECK(zero.moveCount() == 0);
  CHECK(zero.lastMove() == kNoMove);
  for (Move m : {kPass, static_cast<Move>(0)}) {
    Board one(2, 0.0f, 1);
    CHECK_FALSE(one.gameOver());
    one.play(m);
    CHECK(one.gameOver());
    CHECK(one.moveCount() == 1);
    CHECK(one.lastMove() == m);
  }
  Board b(5, 7.5f, 3);
  b.play(0);
  const uint64_t hash = b.hash();
  const Snapshot stones = b.snapshot(0);
  for (int passes = 1; passes <= 2; ++passes) {
    b.play(kPass);
    CHECK(b.hash() == hash);
    CHECK(b.snapshot(0) == stones);
    CHECK(b.consecutivePasses() == passes);
    CHECK(b.moveCount() == 1 + passes);
    CHECK(b.lastMove() == kPass);
    CHECK(b.toMove() == (passes == 1 ? Color::Black : Color::White));
    CHECK(b.gameOver() == (passes == 2));
  }
}

TEST_CASE("empty and white-only area scoring and zero-komi draws") {
  for (int n : {2, 19}) {
    Board empty(n, 7.5f);
    std::vector<int8_t> owner(n * n, 42);
    empty.areaOwnership(owner.data());
    CHECK(std::all_of(owner.begin(), owner.end(), [](int8_t o) { return o == 0; }));
    CHECK(empty.score() == doctest::Approx(-7.5f));
    empty.setToMove(Color::White);
    empty.play(0);
    empty.areaOwnership(owner.data());
    CHECK(std::all_of(owner.begin(), owner.end(), [](int8_t o) { return o == -1; }));
    CHECK(empty.score() == doctest::Approx(-n * n - 7.5f));
  }
  Board draw = Board::fromString("X . O\nX . O\nX . O\n", 0.0f);
  CHECK(draw.score() == doctest::Approx(0.0f));
  int8_t owner[9];
  draw.areaOwnership(owner);
  for (int r = 0; r < 3; ++r) {
    CHECK(owner[r * 3] == 1);
    CHECK(owner[r * 3 + 1] == 0);
    CHECK(owner[r * 3 + 2] == -1);
  }
}

TEST_CASE("diagram parsing validates dimensions and accepts whitespace variations") {
  CHECK_THROWS_AS(Board(1, 0.0f), std::invalid_argument);
  CHECK_THROWS_AS(Board(20, 0.0f), std::invalid_argument);
  CHECK_THROWS_AS(Board::fromString("", 0.0f), std::invalid_argument);
  CHECK_THROWS_AS(Board::fromString(".\n", 0.0f), std::invalid_argument);
  CHECK_THROWS_AS(Board::fromString("..\n...\n", 0.0f), std::invalid_argument);
  std::string oversized;
  for (int r = 0; r < 20; ++r) oversized += std::string(20, '.') + '\n';
  CHECK_THROWS_AS(Board::fromString(oversized, 0.0f), std::invalid_argument);

  const Board expected = Board::fromString("X .\n. O\n", 6.5f, Color::White, 10);
  for (const std::string diagram : {"X.\n.O", " \tX .\r\n\t. O\r\n", "\nX .\n\n. O\n\n"}) {
    Board b = Board::fromString(diagram, 6.5f, Color::White, 10);
    CHECK(b == expected);
    CHECK(b.toString() == "X .\n. O\n");
    CHECK(b.moveCount() == 0);
    CHECK(b.lastMove() == kNoMove);
    CHECK(b.snapshotCount() == 1);
    CHECK(b.snapshot(0).stones[0].test(0));
    CHECK(b.snapshot(0).stones[1].test(3));
  }
  // Preserve the existing permissive parser: non-stone characters are ignored.
  CHECK(Board::fromString("[X.]\n[.O]", 6.5f, Color::White, 10) == expected);
}

TEST_CASE("predicted hashes match every locally legal candidate without changing the source") {
  for (int n : {2, 5, 9}) {
    for (uint64_t seed = 41; seed < 44; ++seed) {
      randomGame(n, 7.5f, seed, 0.02, [&](const Board& b, const RefBoard& ref, const GameHistory&) {
        const Board before = b;
        for (int p = -1; p < b.numPoints(); ++p) {
          const Move m = static_cast<Move>(p);
          CAPTURE(n);
          CAPTURE(seed);
          CAPTURE(b.moveCount());
          CAPTURE(p);
          uint64_t predicted;
          std::vector<Color> simulated;
          const bool locallyLegal = b.tryMoveHash(m, &predicted);
          REQUIRE(locallyLegal == ref.simulate(m, simulated));
          if (!locallyLegal) continue;
          Board copy = b;
          copy.play(m);
          CHECK(copy.hash() == predicted);
          for (int q = 0; q < b.numPoints(); ++q) CHECK(copy.atPoint(q) == simulated[q]);
        }
        CHECK(b == before);
        CHECK(b.lastMove() == before.lastMove());
      });
    }
  }
}

TEST_CASE("single stone capture") {
  Board b = Board::fromString(
      ". . . . .\n"
      ". . X . .\n"
      ". X O X .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::Black);
  const int n = 5;
  CHECK(b.liberties(pointOf(2, 2, n)) == 1);
  CHECK(b.chainSize(pointOf(2, 2, n)) == 1);
  GameHistory h;
  h.reset(b.hash());
  REQUIRE(b.isLegal(pt(3, 2, n), HashHistory(h)));
  uint64_t before = b.hash();
  b.play(pt(3, 2, n));
  CHECK(b.get(2, 2) == Color::Empty);
  CHECK(b.get(3, 2) == Color::Black);
  CHECK(b.liberties(pointOf(3, 2, n)) == 4);  // (2,2) (4,2) (3,1) (3,3)
  CHECK(b.hash() != before);
  // Hash equals the hash of the same position rebuilt from scratch.
  Board rebuilt = Board::fromString(b.toString(), 7.5f, Color::White);
  CHECK(rebuilt.hash() == b.hash());
}

TEST_CASE("merging chains dedups shared liberties") {
  const int n = 5;
  Board b(n, 7.5f);
  b.play(pt(1, 1, n));  // B
  b.play(pt(4, 4, n));  // W
  b.play(pt(1, 3, n));  // B
  b.play(pt(4, 3, n));  // W
  CHECK(b.liberties(pointOf(1, 1, n)) == 4);
  CHECK(b.liberties(pointOf(1, 3, n)) == 4);
  b.play(pt(1, 2, n));  // B joins both
  CHECK(b.chainSize(pointOf(1, 2, n)) == 3);
  CHECK(b.liberties(pointOf(1, 1, n)) == 8);
  CHECK(b.liberties(pointOf(1, 3, n)) == 8);
  RefBoard ref(n, 7.5f);
  for (Move m : {pt(1, 1, n), pt(4, 4, n), pt(1, 3, n), pt(4, 3, n), pt(1, 2, n)}) ref.play(m);
  CHECK(ref.liberties(pointOf(1, 2, n)) == 8);
}

TEST_CASE("suicide is illegal, capture is not suicide") {
  const int n = 5;
  Board b = Board::fromString(
      ". X . . .\n"
      "X . X . .\n"
      ". X . . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::White);
  GameHistory h;
  h.reset(b.hash());
  CHECK_FALSE(b.isLegal(pt(1, 1, n), HashHistory(h)));
  uint64_t hh;
  CHECK_FALSE(b.tryMoveHash(pt(1, 1, n), &hh));

  Board c = Board::fromString(
      ". X O . .\n"
      "X . X O .\n"
      ". X O . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::White);
  GameHistory h2;
  h2.reset(c.hash());
  CHECK(c.liberties(pointOf(1, 2, n)) == 1);
  REQUIRE(c.isLegal(pt(1, 1, n), HashHistory(h2)));
  c.play(pt(1, 1, n));
  CHECK(c.get(1, 2) == Color::Empty);
  CHECK(c.get(1, 1) == Color::White);
  CHECK(c.liberties(pointOf(1, 1, n)) == 1);
}

TEST_CASE("simple ko via positional superko") {
  const int n = 5;
  Board b = Board::fromString(
      ". X O . .\n"
      "X O . O .\n"
      ". X O . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::Black);
  GameHistory h;
  h.reset(b.hash());
  REQUIRE(b.isLegal(pt(1, 2, n), HashHistory(h)));
  b.play(pt(1, 2, n));
  h.push(pt(1, 2, n), b.hash());
  CHECK(b.get(1, 1) == Color::Empty);
  // Immediate recapture recreates the initial position: illegal.
  uint64_t hh;
  CHECK(b.tryMoveHash(pt(1, 1, n), &hh));  // not suicide, captures
  CHECK(hh == h.hashes[0]);
  CHECK_FALSE(b.isLegal(pt(1, 1, n), HashHistory(h)));
  // After a ko threat exchange elsewhere, the recapture is legal.
  b.play(pt(4, 4, n));
  h.push(pt(4, 4, n), b.hash());
  b.play(pt(4, 0, n));
  h.push(pt(4, 0, n), b.hash());
  CHECK(b.isLegal(pt(1, 1, n), HashHistory(h)));
}

TEST_CASE("pass followed by a non-capturing move can repeat a position (must be illegal)") {
  const int n = 5;
  // Black chain {(0,1),(0,2)} has the single liberty (0,0). White captures it by
  // playing (0,0); that stone is then in atari at (0,1). Black recaptures with (0,1),
  // White passes, and Black's non-capturing (0,2) would recreate the initial position.
  Board b = Board::fromString(
      ". X X O .\n"
      "X O O . .\n"
      ". . . . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::White);
  REQUIRE(b.liberties(pointOf(0, 1, n)) == 1);
  RefBoard ref(n, 7.5f);
  ref.cells.assign(n * n, Color::Empty);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) ref.cells[r * n + c] = b.get(r, c);
  ref.toMove = Color::White;
  ref.positions = {ref.cells};

  GameHistory h;
  h.reset(b.hash());
  const uint64_t initial = b.hash();
  auto playBoth = [&](Move m) {
    REQUIRE(b.isLegal(m, HashHistory(h)));
    REQUIRE(ref.isLegal(m));
    b.play(m);
    ref.play(m);
    h.push(m, b.hash());
  };
  playBoth(pt(0, 0, n));  // W captures the two black stones
  CHECK(b.get(0, 1) == Color::Empty);
  CHECK(b.get(0, 2) == Color::Empty);
  CHECK(b.liberties(pointOf(0, 0, n)) == 1);
  playBoth(pt(0, 1, n));  // B captures the white stone at (0,0)
  CHECK(b.get(0, 0) == Color::Empty);
  playBoth(kPass);        // W passes
  // B at (0,2) captures nothing but recreates the initial position.
  uint64_t hh;
  REQUIRE(b.tryMoveHash(pt(0, 2, n), &hh));
  CHECK(hh == initial);
  CHECK_FALSE(b.isLegal(pt(0, 2, n), HashHistory(h)));
  CHECK_FALSE(ref.isLegal(pt(0, 2, n)));
  // Sanity: it would be legal if the history were forgotten.
  GameHistory fresh;
  fresh.reset(b.hash());
  CHECK(b.isLegal(pt(0, 2, n), HashHistory(fresh)));
}

TEST_CASE("game ends after two passes or at the move cap") {
  Board b(5, 7.5f);
  b.play(kPass);
  CHECK_FALSE(b.gameOver());
  b.play(static_cast<Move>(12));
  CHECK(b.consecutivePasses() == 0);
  b.play(kPass);
  b.play(kPass);
  CHECK(b.gameOver());

  Board c(5, 7.5f, 3);
  c.play(static_cast<Move>(0));
  c.play(static_cast<Move>(1));
  CHECK_FALSE(c.gameOver());
  c.play(static_cast<Move>(2));
  CHECK(c.gameOver());
}

TEST_CASE("area scoring: dame and seki-like regions are neutral") {
  Board b = Board::fromString(
      "X X X O O\n"
      "X X X O O\n"
      "X X X O O\n"
      "X X X O O\n"
      "X X X . O\n",
      7.5f);
  CHECK(b.score() == doctest::Approx(15 - 9 - 7.5));
  int8_t owner[25];
  b.areaOwnership(owner);
  CHECK(owner[pointOf(4, 3, 5)] == 0);

  Board s = Board::fromString(
      "X X X X X\n"
      "X . O . X\n"
      "X O O O X\n"
      "X . O . X\n"
      "X X X X X\n",
      7.5f);
  CHECK(s.score() == doctest::Approx(16 - 5 - 7.5));
  s.areaOwnership(owner);
  CHECK(owner[pointOf(1, 1, 5)] == 0);
  CHECK(owner[pointOf(3, 3, 5)] == 0);

  Board t = Board::fromString(
      ". . . . .\n"
      ". X . . .\n"
      ". . . . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f);
  CHECK(t.score() == doctest::Approx(25 - 7.5));  // everything reaches only black
}

TEST_CASE("hash matches a rebuilt position throughout random games") {
  for (int n : {5, 9}) {
    for (uint64_t seed = 1; seed <= 5; ++seed) {
      randomGame(n, 7.5f, seed, 0.02, [&](const Board& b, const RefBoard& ref, const GameHistory&) {
        Board rebuilt = Board::fromString(b.toString(), 7.5f, b.toMove());
        CHECK(rebuilt.hash() == b.hash());
        CHECK(ref.hash() == b.hash());
      });
    }
  }
}

TEST_CASE("differential fuzz against the reference board") {
  struct Cfg {
    int n;
    int games;
    int sampleEvery;  // check all points every k-th move, otherwise a sample
  };
  const Cfg cfgs[] = {{2, 100, 1}, {5, 400, 1}, {7, 150, 1}, {9, 80, 1}, {13, 10, 5}, {19, 4, 20}};
  const int scale = fuzzScale();
  for (const Cfg& cfg : cfgs) {
    for (int g = 0; g < cfg.games * scale; ++g) {
      const uint64_t seed = 1000 * cfg.n + g;
      Rng sampler(seed ^ 0xABCDEF);
      randomGame(cfg.n, 7.5f, seed, 0.03, [&](const Board& b, const RefBoard& ref, const GameHistory& h) {
        const int nn = cfg.n * cfg.n;
        REQUIRE(b.toMove() == ref.toMove);
        REQUIRE(b.moveCount() == ref.moveCount);
        REQUIRE(b.hash() == ref.hash());
        for (int p = 0; p < nn; ++p) REQUIRE(b.atPoint(p) == ref.cells[p]);
        const bool full = (b.moveCount() % cfg.sampleEvery) == 0;
        for (int i = 0; i < (full ? nn : 24); ++i) {
          int p = full ? i : static_cast<int>(sampler.uniformInt(static_cast<uint32_t>(nn)));
          REQUIRE(b.isLegal(static_cast<Move>(p), HashHistory(h)) == ref.isLegal(static_cast<Move>(p)));
          REQUIRE(b.liberties(p) == ref.liberties(p));
          REQUIRE(b.chainSize(p) == ref.chainSize(p));
        }
        if (b.gameOver()) REQUIRE(b.score() == doctest::Approx(ref.score()));
      });
    }
  }
}
