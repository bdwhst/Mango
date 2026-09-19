#include <memory>

#include "doctest/doctest.h"
#include "gtp/gtp.h"
#include "gtp/gtp_options.h"

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

TEST_CASE("gtp komi replays the game with the original colours") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 9, 7.5f);
  bool quit;
  CHECK(e.handle("play b D4", &quit) == "=");
  CHECK(e.handle("play b F6", &quit) == "=");  // out of turn
  CHECK(e.handle("play w E5", &quit) == "=");
  CHECK(e.handle("komi 0.5", &quit) == "=");
  Move d4, f6, e5;
  GtpEngine::parseVertex("D4", 9, &d4);
  GtpEngine::parseVertex("F6", 9, &f6);
  GtpEngine::parseVertex("E5", 9, &e5);
  CHECK(e.board().atPoint(d4) == Color::Black);
  CHECK(e.board().atPoint(f6) == Color::Black);
  CHECK(e.board().atPoint(e5) == Color::White);
  CHECK(e.board().toMove() == Color::Black);  // white moved last
  CHECK(e.board().moveCount() == 3);
  CHECK(e.history().size() == 3);
  CHECK(e.board().komi() == doctest::Approx(0.5f));
  // Three stones, every empty region touches both colours: 2 - 1 - 0.5.
  CHECK(e.board().score() == doctest::Approx(0.5f));
  CHECK(e.handle("final_score", &quit) == "= B+0.5");
  // A negative komi is a value, not a sentinel.
  CHECK(e.handle("komi -3.5", &quit) == "=");
  CHECK(e.board().score() == doctest::Approx(4.5f));
  CHECK(e.board().toMove() == Color::Black);
}

TEST_CASE("gtp komi restriction: a negative model komi is still enforced") {
  GtpEngine e(std::make_unique<RandomPlayer>(1), 9, -2.5f, -2.5f);
  bool quit;
  CHECK(e.handle("komi -2.5", &quit) == "=");
  CHECK(e.handle("komi 7.5", &quit).rfind("? unsupported komi", 0) == 0);
  CHECK(e.board().komi() == doctest::Approx(-2.5f));
  GtpEngine any(std::make_unique<RandomPlayer>(1), 9, -2.5f, std::nullopt);
  CHECK(any.handle("komi 7.5", &quit) == "=");
}

TEST_CASE("gtp options: parsing") {
  GtpOptions o;
  std::string err;
  const char* argv1[] = {"mango_gtp", "--size", "5", "--komi", "-7.5", "--seed", "42", "--model", "m", "--sims", "10",
                         "--device", "cpu", "--config", "c.json", "--fp32", "--allow-komi-mismatch"};
  REQUIRE(GtpOptions::parse(17, argv1, &o, &err));
  CHECK(o.size == 5);
  REQUIRE(o.komi.has_value());
  CHECK(*o.komi == doctest::Approx(-7.5f));
  CHECK(o.seed == 42);
  CHECK(o.model == "m");
  CHECK(o.sims == 10);
  CHECK(o.device == "cpu");
  CHECK(o.configPath == "c.json");
  CHECK(o.fp32);
  CHECK(o.allowKomiMismatch);
  CHECK_FALSE(o.help);

  const char* argv2[] = {"mango_gtp"};
  REQUIRE(GtpOptions::parse(1, argv2, &o, &err));
  CHECK_FALSE(o.komi.has_value());
  CHECK(o.size == -1);
  CHECK(o.device == "auto");
  CHECK_FALSE(o.allowKomiMismatch);
  const char* argv3[] = {"mango_gtp", "--komi"};
  CHECK_FALSE(GtpOptions::parse(2, argv3, &o, &err));
  CHECK(err == "missing value for --komi");
  const char* argv4[] = {"mango_gtp", "--bogus"};
  CHECK_FALSE(GtpOptions::parse(2, argv4, &o, &err));
  CHECK(err == "unknown option --bogus");
  const char* argv5[] = {"mango_gtp", "-h"};
  REQUIRE(GtpOptions::parse(2, argv5, &o, &err));
  CHECK(o.help);
}

TEST_CASE("gtp options: size, komi and model-komi precedence") {
  const Config base = Config::fromJsonText(R"({"board": {"size": 9, "komi": 6.5}, "search": {"eval_simulations": 50}})");
  const ModelMeta meta = ModelMeta::fromJsonText(
      R"({"format": "torchscript-v1", "model_id": "0001-deadbeef", "board_size": 5, "planes": 17,
          "feature_schema": 1, "rules_id": 1, "komi": 7.5, "move_cap": 50, "res_blocks": 1, "filters": 8})");
  GtpOptions o;

  SUBCASE("random player: config values, komi command unrestricted") {
    GtpSetup s = resolveGtpSetup(o, base, nullptr);
    CHECK(s.config.board.size == 9);
    CHECK(s.config.board.komi == doctest::Approx(6.5f));
    CHECK(s.config.search.evalSimulations == 50);
    CHECK_FALSE(s.modelKomi.has_value());
  }
  SUBCASE("model: size and komi come from the model, komi command pinned to it") {
    GtpSetup s = resolveGtpSetup(o, base, &meta);
    CHECK(s.config.board.size == 5);
    CHECK(s.config.board.komi == doctest::Approx(7.5f));
    REQUIRE(s.modelKomi.has_value());
    CHECK(*s.modelKomi == doctest::Approx(7.5f));
  }
  SUBCASE("--komi different from the model is an error unless --allow-komi-mismatch") {
    o.komi = 6.5f;
    CHECK_THROWS(resolveGtpSetup(o, base, &meta));
    o.allowKomiMismatch = true;
    GtpSetup s = resolveGtpSetup(o, base, &meta);
    CHECK(s.config.board.komi == doctest::Approx(6.5f));
    CHECK_FALSE(s.modelKomi.has_value());  // the GTP "komi" command is unrestricted too
  }
  SUBCASE("a negative --komi is honoured, not treated as 'not given'") {
    o.komi = -7.5f;
    CHECK_THROWS(resolveGtpSetup(o, base, &meta));
    o.allowKomiMismatch = true;
    CHECK(resolveGtpSetup(o, base, &meta).config.board.komi == doctest::Approx(-7.5f));
    CHECK(resolveGtpSetup(o, base, nullptr).config.board.komi == doctest::Approx(-7.5f));
  }
  SUBCASE("--komi equal to the model's needs no flag and keeps the restriction") {
    o.komi = 7.5f;
    GtpSetup s = resolveGtpSetup(o, base, &meta);
    REQUIRE(s.modelKomi.has_value());
    CHECK(*s.modelKomi == doctest::Approx(7.5f));
  }
  SUBCASE("--size overrides; a size the model cannot play is rejected; --sims overrides") {
    o.size = 7;
    CHECK_THROWS(resolveGtpSetup(o, base, &meta));
    CHECK(resolveGtpSetup(o, base, nullptr).config.board.size == 7);
    o.sims = 10;
    CHECK(resolveGtpSetup(o, base, nullptr).config.search.evalSimulations == 10);
  }
}
