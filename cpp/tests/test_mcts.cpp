// Sequential search tests (docs/DESIGN.md section 9, "Search, sequential"), all with
// FakeEvaluator, symmetry off, noise off, cache off.
#include <cmath>
#include <map>
#include <stdexcept>

#include "core/board.h"
#include "core/features.h"
#include "doctest/doctest.h"
#include "nn/evaluator.h"
#include "search/mcts.h"

using namespace mango;

namespace {

SearchParams testParams(int sims) {
  SearchParams p;
  p.simulations = sims;
  p.cPuct = 1.5f;
  p.fpu = SearchParams::Fpu::Zero;
  p.addRootNoise = false;
  p.temperatureMoves = 0;
  p.searchSymmetry = false;
  p.treeReuse = true;
  p.resignThreshold = -1.0f;
  return p;
}

bool blackToMove(const uint8_t* planes, int n) { return planes[16 * n * n] == 1; }

// Value from the perspective of the player to move, as a function of the position.
FakeEvaluator colorValueEvaluator(int n, float valueForBlack) {
  return FakeEvaluator(n, [n, valueForBlack](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    *value = blackToMove(planes, n) ? valueForBlack : -valueForBlack;
  });
}

Move pt(int r, int c, int n) { return static_cast<Move>(pointOf(r, c, n)); }

bool occupied(const uint8_t* planes, int n, int p) { return planes[p] == 1 || planes[8 * n * n + p] == 1; }
bool anyStone(const uint8_t* planes, int n) {
  for (int p = 0; p < n * n; ++p)
    if (occupied(planes, n, p)) return true;
  return false;
}

// Evaluator that steers a normal search along `line`: at any position the first line
// point that is still empty gets an overwhelming prior (pass entries in `line` mean
// "prefer pass once everything before it is on the board"). Value 0 everywhere, so
// only priors decide; with FPU Q = 0 the search then descends the line one ply per
// simulation through the production collect/commit path.
FakeEvaluator lineEvaluator(int n, std::vector<Move> line) {
  return FakeEvaluator(n, [n, line](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    for (Move m : line) {
      if (m == kPass) {
        policy[n * n] = 1000.0f;
        break;
      }
      if (!occupied(planes, n, m)) {
        policy[m] = 1000.0f;
        break;
      }
    }
    *value = 0.0f;
  });
}

// Read-only walk: follows the edges for `moves` from the root; nullptr if an edge is
// missing or its child has not been expanded by the search.
const Node* walk(const Node& root, const std::vector<Move>& moves) {
  const Node* nd = &root;
  for (Move m : moves) {
    const Edge* found = nullptr;
    for (const Edge& e : nd->edges)
      if (e.move == m) found = &e;
    if (!found || !found->child) return nullptr;
    nd = found->child.get();
  }
  return nd;
}

}  // namespace

TEST_CASE("first selection at a fresh node is the highest prior") {
  const int n = 5;
  FakeEvaluator ev(n, [n](const uint8_t*, float* policy, float* value) {
    for (int a = 0; a <= n * n; ++a) policy[a] = 0.01f;
    policy[pointOf(2, 2, n)] = 5.0f;
    policy[pointOf(1, 1, n)] = 2.0f;
    *value = 0.0f;
  });
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  SearchTree tree(testParams(1), n, 1);
  tree.newGame(b, h);
  tree.runSequential(ev);
  CHECK(tree.simulationsThisMove() == 1);
  CHECK(tree.rootTotalVisits() == 1);
  CHECK(tree.root().visitCount == 2);
  const Node& r = tree.root();
  const Edge* visited = nullptr;
  for (const Edge& e : r.edges)
    if (e.N == 1) visited = &e;
  REQUIRE(visited != nullptr);
  CHECK(visited->move == pointOf(2, 2, n));
  CHECK(tree.selectEdge(r) >= 0);
  CHECK(tree.checkInvariants() == "");
  // Ties (uniform prior, all N = 0) resolve to the lowest index.
  FakeEvaluator uni(n);
  SearchTree t2(testParams(1), n, 1);
  t2.newGame(b, h);
  t2.runSequential(uni);
  for (const Edge& e : t2.root().edges)
    if (e.N == 1) CHECK(e.move == 0);
}

TEST_CASE("backup signs: Q is from the perspective of the player choosing the edge") {
  const int n = 5;
  FakeEvaluator ev = colorValueEvaluator(n, 0.5f);  // black-to-move positions: +0.5, white: -0.5
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  // Plain search (5.4.4): with game-ending moves resolved (5.4.10) a pass answered by a
  // pass carries an exact value that would mix into these constant-value averages.
  SearchParams p = testParams(60);
  p.resolveTerminalMoves = false;
  SearchTree tree(p, n, 3);
  tree.newGame(b, h);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  CHECK(tree.rootValue() == doctest::Approx(0.5f));
  int depth1 = 0, depth2 = 0;
  for (const Edge& e : tree.root().edges) {
    if (e.N == 0) continue;
    ++depth1;
    CHECK(e.Q() == doctest::Approx(0.5f));  // black chooses, black is winning
    if (e.child && e.child->state == NodeState::Expanded) {
      CHECK(e.child->nnValue == doctest::Approx(-0.5f));  // white to move at the child
      for (const Edge& g : e.child->edges) {
        if (g.N == 0) continue;
        ++depth2;
        CHECK(g.Q() == doctest::Approx(-0.5f));  // white chooses, white is losing
      }
    }
  }
  CHECK(depth1 > 0);
  CHECK(depth2 > 0);
  CHECK(tree.rootMaxQ() == doctest::Approx(0.5f));
}

TEST_CASE("terminal leaves use the exact result with the correct sign") {
  const int n = 5;
  // Black owns the board; white to move after black passed. Passing ends the game.
  Board b = Board::fromString(
      ". . . . .\n"
      ". X X X .\n"
      ". X . X .\n"
      ". X X X .\n"
      ". . . . .\n",
      7.5f, Color::Black);
  b.play(kPass);  // white to move, one pass on record
  GameHistory h;
  h.reset(b.hash());
  h.push(kPass, b.hash());
  FakeEvaluator ev(n, 0.0f);
  SearchTree tree(testParams(200), n, 5);
  tree.newGame(b, h);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  const Edge* pass = nullptr;
  for (const Edge& e : tree.root().edges)
    if (e.move == kPass) pass = &e;
  REQUIRE(pass != nullptr);
  CHECK(pass->N > 0);
  CHECK(pass->Q() == doctest::Approx(-1.0f));  // white passes -> game over -> black wins
  REQUIRE(pass->child != nullptr);
  CHECK(pass->child->state == NodeState::Terminal);
  CHECK(pass->child->terminalValue == doctest::Approx(1.0f));  // perspective of black (to move at the terminal)
  CHECK(pass->child->visitCount == pass->N);

  // Move cap: a board with cap 1 ends after any move; white is to move and loses.
  Board c = Board::fromString(
      ". . . . .\n"
      ". X X X .\n"
      ". X . X .\n"
      ". X X X .\n"
      ". . . . .\n",
      7.5f, Color::White, /*moveCap=*/1);
  GameHistory h2;
  h2.reset(c.hash());
  SearchTree t2(testParams(30), n, 5);
  t2.newGame(c, h2);
  const int positionsBefore = ev.positions();  // ev is shared with the first tree
  t2.runSequential(ev);
  REQUIRE(t2.checkInvariants() == "");
  for (const Edge& e : t2.root().edges)
    if (e.N > 0) CHECK(e.Q() == doctest::Approx(-1.0f));
  // Only the root was evaluated: every child is terminal, so no other position reaches the net.
  CHECK(ev.positions() - positionsBefore == 1);
  CHECK(t2.terminalSimulationsThisMove() == 30);  // all 30 simulations ended at a terminal child
}

TEST_CASE("PUCT prefers higher Q at equal N and higher prior at equal Q") {
  const int n = 5;
  SearchTree tree(testParams(1), n, 1);
  Node node;
  node.state = NodeState::Expanded;
  node.nnValue = 0.0f;
  node.visitCount = 1;
  for (int i = 0; i < 3; ++i) {
    Edge e;
    e.move = static_cast<Move>(i);
    e.P = 0.2f;
    node.edges.push_back(std::move(e));
  }
  node.edges[0].P = 0.6f;
  CHECK(tree.selectEdge(node) == 0);  // prior decides at N = 0
  node.edges[0].N = node.edges[1].N = node.edges[2].N = 4;
  node.edges[0].W = 0.0f;
  node.edges[1].W = 2.0f;  // Q = 0.5
  node.edges[2].W = 1.0f;
  node.visitCount = 13;
  CHECK(tree.selectEdge(node) == 1);
  node.edges[2].W = 2.0f;
  node.edges[2].P = 0.9f;
  CHECK(tree.selectEdge(node) == 2);  // equal Q, higher prior
}

TEST_CASE("search converges on the only winning move in a 3-ply ladder") {
  const int n = 5;
  // White chain {(1,1),(1,2)} has liberties (1,3) and (2,1). Black (1,3) is the only
  // winning atari: white must extend to (2,1), which is still in atari at (3,1), and
  // black captures. Black (2,1) instead lets white extend to (1,3) with two liberties.
  // The game is capped 3 plies from now. With komi 8.5: capture after the extension
  // scores +16.5, capture after a white tenuki +1.5, any line where the chain lives
  // (white escapes with two liberties) at most -1.5. So only (1,3) wins.
  Board b = Board::fromString(
      ". X X X .\n"
      "X O O . .\n"
      "X . X . .\n"
      ". . . . .\n"
      ". . . . .\n",
      8.5f, Color::Black, /*moveCap=*/3);
  GameHistory h;
  h.reset(b.hash());
  FakeEvaluator ev(n, 0.0f);  // uniform priors, zero values: only terminal results carry signal
  // The rule of 5.4.10 resolves every move at the cap exactly; the search then still
  // converges on the winning atari (checked first), but the detailed Q expectations
  // below describe the plain search of 5.4.4, so the main run switches the rule off.
  {
    SearchParams p = testParams(3000);
    p.resolveTerminalMoves = true;
    SearchTree resolved(p, n, 9);
    resolved.newGame(b, h);
    resolved.runSequential(ev);
    REQUIRE(resolved.checkInvariants() == "");
    CHECK(resolved.selectMove(100) == pt(1, 3, n));
    std::vector<float> piR;
    resolved.policyTarget(piR);
    CHECK(piR[pt(1, 3, n)] > 0.5f);
    for (const Edge& e : resolved.root().edges)
      if (e.move == pt(1, 3, n)) CHECK(e.Q() > 0.9f);
  }
  SearchParams plain = testParams(3000);
  plain.resolveTerminalMoves = false;
  SearchTree tree(plain, n, 9);
  tree.newGame(b, h);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  Move m = tree.selectMove(100);
  CHECK(m == pt(1, 3, n));
  std::vector<float> pi;
  tree.policyTarget(pi);
  double sum = 0.0;
  for (float x : pi) sum += x;
  CHECK(sum == doctest::Approx(1.0));
  CHECK(pi[pt(1, 3, n)] > 0.5f);
  CHECK(static_cast<int>(pi.size()) == n * n + 1);
  // Per action: pi(a) = N(a) / sum N, zero for actions that are not root edges (a one-hot
  // on the most visited move would also pass the three checks above).
  {
    std::map<int, uint32_t> visits;
    uint32_t total = 0;
    for (const Edge& e : tree.root().edges) {
      visits[e.move == kPass ? n * n : e.move] = e.N;
      total += e.N;
    }
    REQUIRE(total == tree.rootTotalVisits());
    int nonzero = 0;
    for (int a = 0; a <= n * n; ++a) {
      const uint32_t na = visits.count(a) ? visits[a] : 0;
      CHECK(pi[a] == doctest::Approx(static_cast<double>(na) / total));
      nonzero += na > 0;
    }
    CHECK(nonzero >= 3);
  }
  float bestQ = -2.0f;
  for (const Edge& e : tree.root().edges)
    if (e.move == pt(1, 3, n)) bestQ = e.Q();
  CHECK(bestQ > 0.9f);
  for (const Edge& e : tree.root().edges) {
    // Losing alternatives are visited ~20 times each and mix unresolved (0) and lost
    // (-1) leaves; they must be clearly worse than the winning move and negative.
    if (e.move != pt(1, 3, n) && e.N >= 10) {
      CHECK(e.Q() < 0.0f);
      CHECK(e.Q() < bestQ - 0.5f);
    }
  }
  // The refutation of the wrong atari is in the tree: after B(2,1), W(1,3) leaves no
  // winning black move, so that white edge has a high Q (white's perspective).
  const Node* afterWrong = tree.forceLine({pt(2, 1, n)}, ev);
  REQUIRE(afterWrong != nullptr);
  bool sawEscape = false;
  for (const Edge& e : afterWrong->edges)
    if (e.move == pt(1, 3, n) && e.N >= 5) {
      sawEscape = true;
      CHECK(e.Q() > 0.7f);
    }
  CHECK(sawEscape);
  const Node* refuted = tree.forceLine({pt(2, 1, n), pt(1, 3, n)}, ev);
  REQUIRE(refuted != nullptr);
  for (const Edge& e : refuted->edges)
    if (e.N > 0) CHECK(e.Q() == doctest::Approx(-1.0f));
}

TEST_CASE("tree reuse keeps the chosen subtree and frees the rest") {
  const int n = 5;
  FakeEvaluator ev(n, 0.1f);
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  SearchParams p = testParams(50);
  SearchTree tree(p, n, 11);
  tree.newGame(b, h);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  CHECK(tree.rootTotalVisits() == 50);
  const int64_t before = Node::liveCount();
  Move m = tree.selectMove(100);
  uint32_t edgeN = 0;
  uint32_t childVisits = 0;
  for (const Edge& e : tree.root().edges)
    if (e.move == m) {
      edgeN = e.N;
      childVisits = e.child ? e.child->visitCount : 0;
    }
  REQUIRE(edgeN > 0);
  b.play(m);
  h.push(m, b.hash());
  tree.advance(m, b, h);
  CHECK(tree.root().visitCount == childVisits);
  CHECK(tree.inheritedVisits() == edgeN - 1);
  CHECK(tree.rootTotalVisits() == edgeN - 1);
  CHECK(Node::liveCount() < before);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  CHECK(tree.simulationsThisMove() == 50);
  CHECK(tree.rootTotalVisits() == tree.inheritedVisits() + 50);

  // Without tree reuse the root is fresh and visits == S exactly.
  p.treeReuse = false;
  SearchTree t2(p, n, 11);
  Board b2(n, 7.5f);
  GameHistory h2;
  h2.reset(b2.hash());
  t2.newGame(b2, h2);
  t2.runSequential(ev);
  Move m2 = t2.selectMove(100);
  b2.play(m2);
  h2.push(m2, b2.hash());
  t2.advance(m2, b2, h2);
  CHECK(t2.inheritedVisits() == 0);
  t2.runSequential(ev);
  CHECK(t2.rootTotalVisits() == 50);
}

TEST_CASE("resignation uses the root player's perspective") {
  const int n = 5;
  FakeEvaluator ev = colorValueEvaluator(n, -0.95f);  // black is losing everywhere
  SearchParams p = testParams(20);
  p.resignThreshold = -0.9f;
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  SearchTree tree(p, n, 2);
  tree.newGame(b, h);
  tree.runSequential(ev);
  CHECK(tree.rootValue() == doctest::Approx(-0.95f));
  CHECK(tree.rootMaxQ() == doctest::Approx(-0.95f));
  CHECK(tree.shouldResign());
  // White to move: same evaluator says white is winning -> no resignation.
  Board w(n, 7.5f);
  w.play(static_cast<Move>(12));
  GameHistory hw;
  hw.reset(w.hash());
  SearchTree t2(p, n, 2);
  t2.newGame(w, hw);
  t2.runSequential(ev);
  CHECK(t2.rootValue() == doctest::Approx(0.95f));
  CHECK_FALSE(t2.shouldResign());
  // Threshold -1 disables resignation.
  p.resignThreshold = -1.0f;
  SearchTree t3(p, n, 2);
  t3.newGame(b, h);
  t3.runSequential(ev);
  CHECK_FALSE(t3.shouldResign());
}

TEST_CASE("resignation needs both the root value and the best Q below the threshold") {
  // Values by depth, chosen so that every value backed up into a root edge has the same
  // sign: a node at odd depth belongs to the opponent (its value is negated once on
  // the way to the root edge), a node at even depth >= 2 to the root player. The root is
  // the only position without stones (empty 5x5), black to move.
  const int n = 5;
  auto depthEvaluator = [n](float rootV, float edgeQ) {
    return FakeEvaluator(n, [n, rootV, edgeQ](const uint8_t* planes, float* policy, float* value) {
      for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
      if (!anyStone(planes, n)) *value = rootV;                    // root
      else if (blackToMove(planes, n)) *value = edgeQ;              // even depth: root player
      else *value = -edgeQ;                                         // odd depth: opponent
    });
  };
  SearchParams p = testParams(20);
  p.resignThreshold = -0.9f;
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());

  SUBCASE("root value low, best Q high: no resignation") {
    FakeEvaluator ev = depthEvaluator(-0.95f, +0.95f);
    SearchTree tree(p, n, 2);
    tree.newGame(b, h);
    tree.runSequential(ev);
    CHECK(tree.rootValue() == doctest::Approx(-0.95f));
    CHECK(tree.rootMaxQ() == doctest::Approx(0.95f));
    CHECK_FALSE(tree.shouldResign());
  }
  SUBCASE("root value high, best Q low: no resignation") {
    FakeEvaluator ev = depthEvaluator(+0.95f, -0.95f);
    SearchTree tree(p, n, 2);
    tree.newGame(b, h);
    tree.runSequential(ev);
    CHECK(tree.rootValue() == doctest::Approx(0.95f));
    CHECK(tree.rootMaxQ() == doctest::Approx(-0.95f));
    CHECK_FALSE(tree.shouldResign());
  }
  SUBCASE("both low: resignation") {
    FakeEvaluator ev = depthEvaluator(-0.95f, -0.95f);
    SearchTree tree(p, n, 2);
    tree.newGame(b, h);
    tree.runSequential(ev);
    CHECK(tree.rootMaxQ() == doctest::Approx(-0.95f));
    CHECK(tree.shouldResign());
  }
  SUBCASE("both just above the threshold: no resignation") {
    FakeEvaluator ev = depthEvaluator(-0.85f, -0.85f);
    SearchTree tree(p, n, 2);
    tree.newGame(b, h);
    tree.runSequential(ev);
    CHECK_FALSE(tree.shouldResign());
  }
}

TEST_CASE("superko-illegal moves are never expanded, including along the search path") {
  const int n = 5;
  // Black has just captured the ko at (1,2); white to move. Retaking at (1,1) is illegal now.
  Board b = Board::fromString(
      ". X O . .\n"
      "X O . O .\n"
      ". X O . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::Black);
  GameHistory h;
  h.reset(b.hash());
  b.play(pt(1, 2, n));
  h.push(pt(1, 2, n), b.hash());
  REQUIRE(b.toMove() == Color::White);
  FakeEvaluator ev(n, 0.0f);
  SearchTree tree(testParams(10), n, 4);
  tree.newGame(b, h);
  tree.runSequential(ev);
  for (const Edge& e : tree.root().edges) CHECK(e.move != pt(1, 1, n));
  CHECK(tree.checkInvariants() == "");

  // Along a line reached by the ordinary search (collect/commit, path hashes built by
  // the production code, not by forceLine): W threat (4,4), B answers (4,0), W retakes
  // (1,1) (legal: new position), then B retaking (1,2) would recreate the position
  // after (4,0) -> that edge must be absent at the node the search expanded.
  {
    const std::vector<Move> line{pt(4, 4, n), pt(4, 0, n), pt(1, 1, n)};
    FakeEvaluator steer = lineEvaluator(n, line);
    SearchTree t2(testParams(12), n, 4);
    t2.newGame(b, h);
    t2.runSequential(steer);
    REQUIRE(t2.checkInvariants() == "");
    const Node* nd = walk(t2.root(), line);
    REQUIRE(nd != nullptr);
    REQUIRE(nd->state == NodeState::Expanded);
    CHECK(nd->toMove == Color::Black);
    bool hasRetake = false;
    for (const Edge& e : nd->edges) hasRetake |= (e.move == pt(1, 2, n));
    CHECK_FALSE(hasRetake);
    // The retake is absent for the superko reason only: every other empty point is an edge.
    int emptyPoints = 0;
    for (int q = 0; q < n * n; ++q) {
      bool onBoard = false;
      for (Move m : {pt(0, 1, n), pt(0, 2, n), pt(1, 0, n), pt(1, 3, n), pt(2, 1, n), pt(2, 2, n), pt(4, 4, n),
                     pt(4, 0, n), pt(1, 1, n)})
        onBoard |= (m == q);
      emptyPoints += !onBoard;
    }
    CHECK(static_cast<int>(nd->edges.size()) == emptyPoints);  // all empty points minus the retake, plus pass
  }
  // Sanity through the same path: the retake IS legal after B(0,4), W pass.
  {
    const std::vector<Move> line{pt(4, 4, n), pt(4, 0, n), pt(1, 1, n), pt(0, 4, n), kPass};
    FakeEvaluator steer = lineEvaluator(n, line);
    SearchTree t3(testParams(14), n, 4);
    t3.newGame(b, h);
    t3.runSequential(steer);
    REQUIRE(t3.checkInvariants() == "");
    const Node* nd2 = walk(t3.root(), line);
    REQUIRE(nd2 != nullptr);
    REQUIRE(nd2->state == NodeState::Expanded);
    CHECK(nd2->toMove == Color::Black);
    bool hasRetake2 = false;
    for (const Edge& e : nd2->edges) hasRetake2 |= (e.move == pt(1, 2, n));
    CHECK(hasRetake2);
  }
}

TEST_CASE("policy target is the normalised visit vector with pass at the last index") {
  const int n = 5;
  // Position of the superko test after black's capture: white to move, (1,1) is an
  // illegal retake, several points are occupied.
  Board b = Board::fromString(
      ". X O . .\n"
      "X O . O .\n"
      ". X O . .\n"
      ". . . . .\n"
      ". . . . .\n",
      7.5f, Color::Black);
  GameHistory h;
  h.reset(b.hash());
  b.play(pt(1, 2, n));
  h.push(pt(1, 2, n), b.hash());
  // Uniform priors and zero values: 60 simulations over 16 legal edges visit every edge,
  // pass included (a large pass prior would instead drive the search into pass-pass
  // terminals and concentrate all visits there).
  FakeEvaluator ev(n, 0.0f);
  SearchTree tree(testParams(60), n, 5);
  tree.newGame(b, h);
  tree.runSequential(ev);
  REQUIRE(tree.checkInvariants() == "");
  std::vector<float> pi;
  tree.policyTarget(pi);
  REQUIRE(static_cast<int>(pi.size()) == n * n + 1);
  std::map<int, uint32_t> visits;
  uint32_t total = 0;
  for (const Edge& e : tree.root().edges) {
    visits[e.move == kPass ? n * n : e.move] = e.N;
    total += e.N;
  }
  REQUIRE(total == 60);
  int nonzero = 0;
  for (int a = 0; a <= n * n; ++a) {
    const uint32_t na = visits.count(a) ? visits[a] : 0;
    CHECK(pi[a] == doctest::Approx(static_cast<double>(na) / total));
    nonzero += pi[a] > 0.0f;
  }
  CHECK(nonzero >= 3);                 // not a one-hot
  CHECK(pi[n * n] > 0.0f);             // pass visited, at the last index
  CHECK(pi[pt(1, 1, n)] == 0.0f);      // illegal retake
  CHECK(pi[pt(0, 1, n)] == 0.0f);      // occupied
  CHECK(pi[pt(1, 2, n)] == 0.0f);      // occupied (the capturing stone)
}

TEST_CASE("invariants hold after every simulation and evaluations are counted") {
  const int n = 7;
  FakeEvaluator ev = colorValueEvaluator(n, 0.2f);
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  SearchParams p = testParams(1);
  SearchTree tree(p, n, 21);
  tree.newGame(b, h);
  int lastPositions = 0;
  for (int i = 1; i <= 40; ++i) {
    // Run one more simulation each time by raising the budget.
    tree.setSimulations(i);
    tree.runSequential(ev);
    REQUIRE(tree.checkInvariants() == "");
    CHECK(tree.simulationsThisMove() == i);
    CHECK(tree.rootTotalVisits() == static_cast<uint32_t>(i));
    CHECK(ev.positions() >= lastPositions);
    lastPositions = ev.positions();
    // Root expansion + one evaluation per simulation that did not end at a terminal.
    CHECK(ev.positions() == 1 + i - tree.terminalSimulationsThisMove());
  }
  CHECK(ev.positions() <= 41);
}

TEST_CASE("a failed evaluation leaves no pending node, at the root or inside the tree") {
  const int n = 5;
  int failuresLeft = 1;
  FakeEvaluator flaky(n, [&](const uint8_t*, float* policy, float* value) {
    if (failuresLeft > 0) {
      --failuresLeft;
      throw std::runtime_error("simulated evaluator failure");
    }
    for (int a = 0; a <= n * n; ++a) policy[a] = 1.0f;
    *value = 0.0f;
  });
  Board b(n, 7.5f);
  GameHistory h;
  h.reset(b.hash());
  SearchTree tree(testParams(10), n, 7);
  tree.newGame(b, h);
  // Root evaluation fails.
  CHECK_THROWS_AS(tree.runSequential(flaky), std::runtime_error);
  CHECK(tree.root().state == NodeState::Unexpanded);
  CHECK(tree.checkInvariants() == "");
  // Retry succeeds and the tree is fully usable.
  tree.runSequential(flaky);
  CHECK(tree.simulationsThisMove() == 10);
  CHECK(tree.checkInvariants() == "");
  // Failure inside the loop: completed simulations are kept, nothing stays pending.
  failuresLeft = 1;
  tree.setSimulations(20);
  CHECK_THROWS_AS(tree.runSequential(flaky), std::runtime_error);
  CHECK(tree.checkInvariants() == "");
  const int done = tree.simulationsThisMove();
  CHECK(done >= 10);
  CHECK(done < 20);
  tree.runSequential(flaky);
  CHECK(tree.simulationsThisMove() == 20);
  CHECK(tree.checkInvariants() == "");
  // advance() works after a failure (no pending evaluations left behind).
  Move m = tree.selectMove(100);
  b.play(m);
  h.push(m, b.hash());
  CHECK_NOTHROW(tree.advance(m, b, h));
}

// --- game-ending moves (DESIGN 5.4.10, M4 review) --------------------------------------------

namespace {

// Value `valueForBlack` in black's perspective (negated when white is to move); pass
// prior `passBlack` when black is to move and `passWhite` when white is (board moves 1).
FakeEvaluator passPriorEvaluator(int n, float valueForBlack, float passBlack, float passWhite) {
  return FakeEvaluator(n, [=](const uint8_t* planes, float* policy, float* value) {
    for (int a = 0; a < n * n; ++a) policy[a] = 1.0f;
    const bool black = blackToMove(planes, n);
    policy[n * n] = black ? passBlack : passWhite;
    *value = black ? valueForBlack : -valueForBlack;
  });
}

const Edge* rootEdge(const SearchTree& t, Move m) {
  for (const Edge& e : t.root().edges)
    if (e.move == m) return &e;
  return nullptr;
}

}  // namespace

TEST_CASE("a pass that lets the opponent end the game at a loss is never chosen") {
  // Black to move, behind if the game ends now (1 black stone, 2 white stones, komi 7.5;
  // every empty point is neutral). The network believes black is winning (+0.9) and
  // gives black's pass a large prior but white's pass a tiny one: without the exact
  // value of white's answering pass the search sees pass -> "white to move, white
  // losing" (+0.9 for black) and passes; the opponent's pass then ends the game at a loss.
  const int n = 5;
  Board b = Board::fromString("X . . . .\n"
                              ". . . . .\n"
                              ". . O . .\n"
                              ". . . . .\n"
                              ". . . . O\n", 7.5f, Color::Black);
  REQUIRE(b.score() < 0);
  REQUIRE(b.consecutivePasses() == 0);
  GameHistory h;
  h.reset(b.hash());
  FakeEvaluator ev = passPriorEvaluator(n, 0.9f, 100.0f, 0.001f);
  // 20 simulations: fewer than the 25 white replies, so the plain search never reaches
  // white's answering pass (the 9x9 situation: 82 replies, a few dozen visits).
  for (bool resolve : {true, false}) {
    CAPTURE(resolve);
    SearchParams p = testParams(20);
    p.resolveTerminalMoves = resolve;
    SearchTree t(p, n, 1);
    t.newGame(b, h);
    t.runSequential(ev);
    CHECK(t.checkInvariants() == "");
    const Edge* pass = rootEdge(t, kPass);
    REQUIRE(pass != nullptr);
    if (resolve) {
      // The child after black's pass can be ended by white with a win: its value is +1
      // for white, so black's pass edge is exactly -1 from its first visit on.
      REQUIRE(pass->N >= 1);
      CHECK(pass->Q() == doctest::Approx(-1.0f));
      REQUIRE(pass->child != nullptr);
      CHECK(pass->child->nnValue == doctest::Approx(-0.9f));  // the raw network value (white's view) is kept
      CHECK(pass->child->terminalMoveValue == doctest::Approx(1.0f));
      CHECK(pass->child->value() == doctest::Approx(1.0f));
      const Edge* whitePass = nullptr;
      for (const Edge& e : pass->child->edges)
        if (e.move == kPass) whitePass = &e;
      REQUIRE(whitePass != nullptr);
      CHECK(whitePass->terminalValue == doctest::Approx(1.0f));
      CHECK(t.selectMove(100) != kPass);
      // No other root move ends the game: their edges carry no terminal value.
      for (const Edge& e : t.root().edges)
        if (e.move != kPass) CHECK_FALSE(e.endsGame());
      CHECK(t.rootValue() == doctest::Approx(0.9f));  // the root itself has no game-ending move
    } else {
      // The blind spot, kept reproducible: the paper's search passes here.
      CHECK(pass->Q() > 0.5f);
      CHECK(t.selectMove(100) == kPass);
    }
  }
}

TEST_CASE("a winning pass after the opponent's pass is found despite a tiny prior") {
  // White just passed; black is ahead if the game ends (2 black stones, 1 white, komi
  // 0.5). The network thinks black is winning anyway (+0.9 for black, so every board
  // move looks like +0.9) and gives pass a prior of 1e-4: with FPU 0 the plain search
  // never tries the pass; the exact +1 makes it the first and best choice.
  const int n = 5;
  Board b = Board::fromString("X . . . .\n"
                              ". . . . .\n"
                              ". . O . .\n"
                              ". . . . .\n"
                              ". . . . X\n", 0.5f, Color::White);
  b.play(kPass);  // black to move, consecutivePasses == 1
  REQUIRE(b.toMove() == Color::Black);
  REQUIRE(b.consecutivePasses() == 1);
  REQUIRE(b.score() > 0);
  GameHistory h;
  h.reset(b.hash());
  h.push(kPass, b.hash());
  FakeEvaluator ev = passPriorEvaluator(n, 0.9f, 1e-4f, 1e-4f);
  for (bool resolve : {true, false}) {
    CAPTURE(resolve);
    SearchParams p = testParams(30);
    p.resolveTerminalMoves = resolve;
    SearchTree t(p, n, 1);
    t.newGame(b, h);
    t.runSequential(ev);
    CHECK(t.checkInvariants() == "");
    const Edge* pass = rootEdge(t, kPass);
    REQUIRE(pass != nullptr);
    if (resolve) {
      CHECK(pass->terminalValue == doctest::Approx(1.0f));
      CHECK(pass->N > 15);  // the exact +1 beats every network-valued move from the first selection on
      CHECK(pass->Q() == doctest::Approx(1.0f));
      CHECK(t.selectMove(100) == kPass);
      CHECK(t.rootValue() == doctest::Approx(1.0f));   // bound: black can end the game with a win
      CHECK(t.rootNetValue() == doctest::Approx(0.9f));
      CHECK(t.terminalSimulationsThisMove() == static_cast<int>(pass->N));
    } else {
      CHECK(pass->N == 0);  // never explored: FPU 0 below the +0.9 board moves, and a 1e-4 prior
      CHECK(t.selectMove(100) != kPass);
      CHECK(t.rootValue() == doctest::Approx(0.9f));
    }
  }
}

TEST_CASE("every move at the move cap is resolved exactly") {
  // Move cap 3 with two moves played: any move ends the game. Black (1 stone) vs white
  // (1 stone), komi 0.5: a board move makes black's area 2 vs 1 (+0.5, a win), a pass
  // ends it at 1 vs 1 (-0.5, a loss). Uniform priors, value 0.
  const int n = 5;
  Board b(n, 0.5f, /*moveCap=*/3);
  b.play(pt(0, 0, n));
  b.play(pt(4, 4, n));
  REQUIRE(b.moveCount() == 2);
  REQUIRE(!b.gameOver());
  GameHistory h;
  h.reset(b.hash());
  FakeEvaluator ev(n, 0.0f);
  SearchParams p = testParams(40);
  SearchTree t(p, n, 1);
  t.newGame(b, h);
  t.runSequential(ev);
  CHECK(t.checkInvariants() == "");
  for (const Edge& e : t.root().edges) {
    CHECK(e.endsGame());
    CHECK(e.terminalValue == doctest::Approx(e.move == kPass ? -1.0f : 1.0f));
    if (e.N > 0) CHECK(e.Q() == doctest::Approx(e.terminalValue));
  }
  CHECK(t.terminalSimulationsThisMove() == 40);  // every simulation ended at a terminal child
  CHECK(rootEdge(t, kPass)->N == 0);
  CHECK(t.selectMove(100) != kPass);
  CHECK(t.rootValue() == doctest::Approx(1.0f));
  CHECK(t.rootNetValue() == doctest::Approx(0.0f));
}
