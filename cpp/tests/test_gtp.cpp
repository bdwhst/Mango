#include <memory>

#include "doctest/doctest.h"
#include "gtp/gtp.h"

using namespace mango;

TEST_CASE("gtp vertex conversion") {
  Move m;
  REQUIRE(GtpEngine::parseVertex("A1", 9, &m));
  CHECK(m == pointOf(8, 0, 9));
  REQUIRE(GtpEngine::parseVertex("j9", 9, &m));
  CHECK(m == pointOf(0, 8, 9));
  REQUIRE(GtpEngine::parseVertex("T19", 19, &m));
  CHECK(m == pointOf(0, 18, 19));
  CHECK_FALSE(GtpEngine::parseVertex("I3", 9, &m));
  CHECK_FALSE(GtpEngine::parseVertex("K3", 9, &m));
  CHECK_FALSE(GtpEngine::parseVertex("A10", 9, &m));
  REQUIRE(GtpEngine::parseVertex("PASS", 9, &m));
  CHECK(m == kPass);
  for (int p = 0; p < 81; ++p) {
    REQUIRE(GtpEngine::parseVertex(GtpEngine::vertexToString(static_cast<Move>(p), 9), 9, &m));
    CHECK(m == p);
  }
}

TEST_CASE("gtp command loop") {
  GtpEngine e(std::make_unique<RandomPlayer>(7), 9, 7.5f);
  bool quit = false;
  CHECK(e.handle("protocol_version", &quit) == "= 2");
  CHECK(e.handle("12 name", &quit) == "= 12 mango-random");
  CHECK(e.handle("boardsize 5", &quit) == "=");
  CHECK(e.handle("boardsize 25", &quit).rfind("?", 0) == 0);
  CHECK(e.handle("play B C3", &quit) == "=");
  CHECK(e.handle("play W C3", &quit) == "? illegal move");
  CHECK(e.handle("play W pass", &quit) == "=");
  std::string r = e.handle("genmove B", &quit);
  CHECK(r.rfind("= ", 0) == 0);
  CHECK(e.handle("undo", &quit) == "=");
  CHECK(e.handle("undo", &quit) == "=");
  CHECK(e.handle("undo", &quit) == "=");
  CHECK(e.handle("undo", &quit) == "? cannot undo");
  CHECK(e.handle("final_score", &quit) == "= W+7.5");  // empty board: 0 - komi
  CHECK(e.handle("known_command play", &quit) == "= true");
  CHECK(e.handle("known_command foo", &quit) == "= false");
  CHECK(e.handle("bogus", &quit) == "? unknown command");
  CHECK(e.handle("# comment only", &quit).empty());
  CHECK_FALSE(quit);
  CHECK(e.handle("quit", &quit) == "=");
  CHECK(quit);
}

TEST_CASE("gtp final_score reflects komi") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 5, 7.5f);
  bool quit;
  e.handle("play B C3", &quit);
  CHECK(e.handle("final_score", &quit) == "= B+17.5");  // 25 - 7.5
  e.handle("clear_board", &quit);
  CHECK(e.handle("final_score", &quit) == "= W+7.5");   // empty board: 0 - 7.5
}

TEST_CASE("gtp undo rebuilds out-of-turn colours correctly") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 9, 7.5f);
  bool quit;
  CHECK(e.handle("play b D4", &quit) == "=");
  CHECK(e.handle("play b F6", &quit) == "=");  // black twice in a row
  CHECK(e.handle("play w E5", &quit) == "=");
  CHECK(e.handle("undo", &quit) == "=");
  Move d4, f6, e5;
  GtpEngine::parseVertex("D4", 9, &d4);
  GtpEngine::parseVertex("F6", 9, &f6);
  GtpEngine::parseVertex("E5", 9, &e5);
  CHECK(e.board().atPoint(d4) == Color::Black);
  CHECK(e.board().atPoint(f6) == Color::Black);
  CHECK(e.board().atPoint(e5) == Color::Empty);
  CHECK(e.board().toMove() == Color::White);  // after two black moves
  CHECK(e.history().size() == 2);
  // Undo everything and check the board is empty again.
  CHECK(e.handle("undo", &quit) == "=");
  CHECK(e.handle("undo", &quit) == "=");
  CHECK(e.board().hash() == 0);
  CHECK(e.board().moveCount() == 0);
}

TEST_CASE("an illegal play does not change the side to move") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 5, 7.5f);
  bool quit;
  CHECK(e.handle("play b C3", &quit) == "=");
  CHECK(e.board().toMove() == Color::White);
  CHECK(e.handle("play b C3", &quit) == "? illegal move");  // occupied, and for black
  CHECK(e.board().toMove() == Color::White);
  CHECK(e.board().moveCount() == 1);
}

TEST_CASE("gtp komi is rejected when a model komi is fixed") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 9, 7.5f, 7.5f);
  bool quit;
  CHECK(e.handle("komi 7.5", &quit) == "=");
  CHECK(e.handle("komi 6.5", &quit).rfind("? unsupported komi", 0) == 0);
}
