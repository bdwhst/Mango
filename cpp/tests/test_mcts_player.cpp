// MctsPlayer through the GTP engine with a FakeEvaluator.
#include <memory>

#include "doctest/doctest.h"
#include "gtp/gtp.h"
#include "gtp/mcts_player.h"
#include "nn/evaluator.h"

using namespace mango;

namespace {
SearchParams params(int sims) {
  SearchParams p;
  p.simulations = sims;
  p.addRootNoise = false;
  p.temperatureMoves = 0;
  p.searchSymmetry = false;
  p.treeReuse = true;
  return p;
}
}  // namespace

TEST_CASE("mcts player rejects board sizes that do not match the model") {
  FakeEvaluator ev(5, 0.0f);
  auto player = std::make_unique<MctsPlayer>(ev, params(10), 1);
  CHECK(player->acceptsBoardSize(5));
  CHECK_FALSE(player->acceptsBoardSize(7));
  GtpEngine e(std::move(player), 5, 7.5f);
  bool quit;
  CHECK(e.handle("boardsize 7", &quit).rfind("? unacceptable size", 0) == 0);
  CHECK(e.board().size() == 5);
  CHECK(e.handle("boardsize 5", &quit) == "=");
  CHECK(e.handle("genmove b", &quit).rfind("= ", 0) == 0);

  // A player constructed for the wrong size never reaches the evaluator.
  FakeEvaluator ev9(9, 0.0f);
  GtpEngine wrong(std::make_unique<MctsPlayer>(ev9, params(10), 1), 5, 7.5f);
  CHECK(wrong.handle("genmove b", &quit).rfind("? board size 5", 0) == 0);
  CHECK(wrong.handle("mango-analyze", &quit).rfind("? board size 5", 0) == 0);
  CHECK(ev9.positions() == 0);
}

TEST_CASE("mcts player reuses the tree across the opponent's move") {
  FakeEvaluator ev(5, 0.1f);
  auto owned = std::make_unique<MctsPlayer>(ev, params(30), 3);
  MctsPlayer* player = owned.get();
  GtpEngine e(std::move(owned), 5, 7.5f);
  bool quit;
  REQUIRE(e.handle("genmove b", &quit).rfind("= ", 0) == 0);
  REQUIRE(player->tree() != nullptr);
  CHECK(player->tree()->inheritedVisits() == 0);
  // Opponent replies with a move the search has certainly visited: the most visited one.
  const Node& root = player->tree()->root();
  const Edge* best = nullptr;
  for (const Edge& ed : root.edges)
    if (!best || ed.N > best->N) best = &ed;
  REQUIRE(best != nullptr);
  // The GTP layer plays black's move first (advance happens on the next sync), so find
  // white's best reply inside black's chosen subtree instead.
  std::string bm = e.history().moves.empty() ? "" : GtpEngine::vertexToString(e.history().moves.back(), 5);
  const Edge* played = nullptr;
  for (const Edge& ed : root.edges)
    if (GtpEngine::vertexToString(ed.move, 5) == bm) played = &ed;
  REQUIRE(played != nullptr);
  REQUIRE(played->child != nullptr);
  const Edge* reply = nullptr;
  for (const Edge& ed : played->child->edges)
    if (ed.N > 0 && (!reply || ed.N > reply->N)) reply = &ed;
  REQUIRE(reply != nullptr);
  const uint32_t replyChildVisits = reply->N - 1;  // edge N == child visitCount; inherited = child's edge sum
  std::string wm = GtpEngine::vertexToString(reply->move, 5);
  REQUIRE(e.handle("play w " + wm, &quit) == "=");
  REQUIRE(e.handle("genmove b", &quit).rfind("= ", 0) == 0);
  // Two moves were appended (black's and white's): the tree must have been advanced
  // through both, so the new root inherited the reply's subtree.
  CHECK(player->tree()->inheritedVisits() == replyChildVisits);
  CHECK(player->tree()->rootTotalVisits() == replyChildVisits + 30);
  CHECK(player->tree()->checkInvariants() == "");

  // An undo breaks the prefix: the tree is rebuilt, not reused.
  REQUIRE(e.handle("undo", &quit) == "=");
  REQUIRE(e.handle("genmove w", &quit).rfind("= ", 0) == 0);
  CHECK(player->tree()->inheritedVisits() == 0);
}
