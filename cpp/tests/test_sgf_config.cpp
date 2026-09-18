#include "core/config.h"
#include "core/sgf.h"
#include "doctest/doctest.h"

using namespace mango;

TEST_CASE("sgf round trip") {
  SgfGame g;
  g.size = 9;
  g.komi = 7.5f;
  g.moves = {static_cast<Move>(pointOf(2, 3, 9)), kPass, static_cast<Move>(pointOf(8, 8, 9))};
  g.result = "B+3.5";
  g.blackName = "a";
  g.whiteName = "b]c";
  g.comment = "seed=42";
  std::string text = writeSgf(g);
  CHECK(text.find("SZ[9]") != std::string::npos);
  CHECK(text.find(";B[dc];W[];B[ii]") != std::string::npos);
  SgfGame back;
  REQUIRE(parseSgf(text, &back));
  CHECK(back.size == 9);
  CHECK(back.komi == doctest::Approx(7.5f));
  CHECK(back.moves == g.moves);
  CHECK(back.result == "B+3.5");
  CHECK(back.whiteName == "b]c");
  CHECK(back.comment == "seed=42");
  CHECK(resultString(3.5f, false, Color::Black) == "B+3.5");
  CHECK(resultString(-0.5f, false, Color::Black) == "W+0.5");
  CHECK(resultString(0.0f, true, Color::White) == "W+R");
  CHECK(resultString(0.0f, false, Color::Black) == "0");
}

TEST_CASE("sgf parser follows the main line through variations") {
  SgfGame g;
  REQUIRE(parseSgf("(;SZ[9]KM[7.5];B[aa](;W[bb];B[cc])(;W[dd]))", &g));
  CHECK(g.moves == std::vector<Move>{static_cast<Move>(pointOf(0, 0, 9)), static_cast<Move>(pointOf(1, 1, 9)),
                                     static_cast<Move>(pointOf(2, 2, 9))});
  // Nested variations, whitespace, escaped brackets in a skipped branch, pass as "tt".
  REQUIRE(parseSgf("(;SZ[19] ;B[aa]\n(;W[tt](;B[cc]C[main \\] line])(;B[dd]))(;W[ee]C[skip \\) me]))", &g));
  CHECK(g.size == 19);
  REQUIRE(g.moves.size() == 3);
  CHECK(g.moves[1] == kPass);
  CHECK(g.moves[2] == pointOf(2, 2, 19));
  // Malformed input is rejected.
  CHECK_FALSE(parseSgf("(;SZ[9];B[aa]", &g));
  CHECK_FALSE(parseSgf(";B[aa]", &g));
  CHECK_FALSE(parseSgf("(;SZ[9];B[zz])", &g));
  CHECK_FALSE(parseSgf("()", &g));
}

TEST_CASE("config defaults, parsing and fingerprint") {
  Config c = Config::fromJsonText("{}");
  CHECK(c.board.size == 9);
  CHECK(c.search.effectiveDirichletAlpha(9) == doctest::Approx(0.03f * 361.0f / 81.0f));
  CHECK(c.board.effectiveMoveCap() == 162);
  Config d = Config::fromJsonText(R"({"board": {"size": 5, "komi": 7.5}, "search": {"simulations": 50, "c_puct": 1.1}})");
  CHECK(d.board.size == 5);
  CHECK(d.search.simulations == 50);
  CHECK(d.search.cPuct == doctest::Approx(1.1f));
  CHECK(d.fingerprint() != c.fingerprint());
  CHECK(Config::fromJsonText(d.toJsonText()).fingerprint() == d.fingerprint());
  CHECK(d.fingerprint().size() == 16);
  CHECK_THROWS(Config::fromJsonText(R"({"board": {"size": 25}})"));
}
