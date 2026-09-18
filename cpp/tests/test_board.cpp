#include <string>

#include "core/board.h"
#include "core/history.h"
#include "doctest/doctest.h"
#include "tests/test_util.h"

using namespace mango;
using namespace mango::test;

namespace {
Move pt(int r, int c, int n) { return static_cast<Move>(pointOf(r, c, n)); }
}  // namespace

TEST_CASE("empty board basics") {
  for (int n : {5, 9, 13, 19}) {
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
  const Cfg cfgs[] = {{5, 400, 1}, {7, 150, 1}, {9, 80, 1}, {13, 10, 5}, {19, 4, 20}};
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
