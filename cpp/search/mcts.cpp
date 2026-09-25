#include "search/mcts.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "core/features.h"
#include "core/symmetry.h"

namespace mango {

SearchTree::SearchTree(const SearchParams& params, int boardSize, uint64_t seed)
    : params_(params), n_(boardSize), rng_(seed), root_(std::make_unique<Node>()), rootBoard_(boardSize, 7.5f) {}

void SearchTree::resetRoot(const Board& board, const GameHistory& hist) {
  root_ = std::make_unique<Node>();
  root_->toMove = board.toMove();
  rootBoard_ = board;
  hist_ = &hist;
  simulationsThisMove_ = 0;
  terminalSimulationsThisMove_ = 0;
  inheritedVisits_ = 0;
  noiseApplied_ = false;
  pendingCount_ = 0;
}

void SearchTree::newGame(const Board& board, const GameHistory& hist) { resetRoot(board, hist); }

void SearchTree::advance(Move move, const Board& boardAfter, const GameHistory& hist) {
  if (pendingCount_ != 0) throw std::logic_error("advance() with pending evaluations");
  std::unique_ptr<Node> next;
  if (params_.treeReuse && root_->state == NodeState::Expanded) {
    for (Edge& e : root_->edges) {
      if (e.move == move && e.child) {
        next = std::move(e.child);
        break;
      }
    }
  }
  if (next) {
    root_ = std::move(next);  // frees the old root and all siblings
    rootBoard_ = boardAfter;
    hist_ = &hist;
    simulationsThisMove_ = 0;
    terminalSimulationsThisMove_ = 0;
    inheritedVisits_ = root_->state == NodeState::Expanded ? root_->totalEdgeVisits() : 0;
    noiseApplied_ = false;
    pendingCount_ = 0;
  } else {
    resetRoot(boardAfter, hist);
  }
}

// ---------------------------------------------------------------------------
// Selection

int SearchTree::selectEdge(const Node& node) const {
  // Virtual losses count as visits with value -1 for the chooser.
  double sumN = 0.0;
  for (const Edge& e : node.edges) sumN += e.N + e.virtualLoss;
  const double sqrtSum = std::sqrt(sumN);
  const float fpu = params_.fpu == SearchParams::Fpu::Parent ? node.value() : 0.0f;
  int best = -1;
  double bestScore = 0.0;
  float bestP = 0.0f;
  for (int i = 0; i < static_cast<int>(node.edges.size()); ++i) {
    const Edge& e = node.edges[i];
    const uint32_t nEff = e.N + e.virtualLoss;
    // An unvisited move that ends the game has an exact value: use it instead of the
    // FPU value (DESIGN 5.4.10), so a winning pass is found and a losing one avoided.
    const double q = nEff > 0 ? (static_cast<double>(e.W) - e.virtualLoss) / nEff : (e.endsGame() ? e.terminalValue : fpu);
    const double u = params_.cPuct * e.P * sqrtSum / (1.0 + nEff);
    const double s = q + u;
    // Tie-break: higher prior first, then lower index (strict comparisons keep the first).
    if (best < 0 || s > bestScore || (s == bestScore && e.P > bestP)) {
      best = i;
      bestScore = s;
      bestP = e.P;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Expansion and backup

// Exact value of a finished game for the player to move at that position (5.4.1 rule 4).
static float terminalValueOf(const Board& board) {
  const float score = board.score();  // black - white - komi
  float v = score > 0 ? 1.0f : (score < 0 ? -1.0f : 0.0f);
  if (board.toMove() == Color::White) v = -v;
  return v;
}

void SearchTree::makeTerminal(Node& node, const Board& board) {
  node.state = NodeState::Terminal;
  node.toMove = board.toMove();
  node.terminalValue = terminalValueOf(board);
}

void SearchTree::resolveTerminalMoves(Node& node, const Board& board) {
  // Moves that end the game: a second consecutive pass, or any move at the cap. Their
  // outcome is exact and costs no evaluation; the chooser can guarantee the best of them.
  node.terminalMoveValue = Edge::kNoTerminal;
  const bool passEnds = board.consecutivePasses() == 1;
  const bool capEnds = board.moveCount() + 1 >= board.moveCap();
  if (!passEnds && !capEnds) return;
  for (Edge& e : node.edges) {
    if (!capEnds && e.move != kPass) continue;
    Board after = board;
    after.play(e.move);
    if (!after.gameOver()) continue;  // cannot happen; kept as a guard
    e.terminalValue = -terminalValueOf(after);  // rule 3: the chooser's perspective
    if (e.terminalValue > node.terminalMoveValue) node.terminalMoveValue = e.terminalValue;
  }
}

void SearchTree::expandInto(Node& node, const Board& board, const HashHistory& hist, const NNOutput& out, int symmetry) {
  const int nn = n_ * n_;
  const float* policy = out.policy.data();
  if (symmetry != 0) {
    scratchPolicy_.resize(nn + 1);
    transformPolicy(out.policy.data(), scratchPolicy_.data(), n_, inverseSymmetry(symmetry));
    policy = scratchPolicy_.data();
  }
  std::vector<Move> legal;
  board.legalMoves(hist, legal);
  node.edges.clear();
  node.edges.reserve(legal.size());
  double sum = 0.0;
  for (Move m : legal) {
    Edge e;
    e.move = m;
    e.P = policy[m == kPass ? nn : m];
    sum += e.P;
    node.edges.push_back(std::move(e));
  }
  if (sum > 0) {
    for (Edge& e : node.edges) e.P = static_cast<float>(e.P / sum);
  } else {
    for (Edge& e : node.edges) e.P = 1.0f / static_cast<float>(node.edges.size());
  }
  node.toMove = board.toMove();
  node.nnValue = out.value;
  if (params_.resolveTerminalMoves) resolveTerminalMoves(node, board);
  node.visitCount = 1;
  node.state = NodeState::Expanded;
}

void SearchTree::backup(const std::vector<Edge*>& path, const std::vector<Node*>& nodes, Node& leaf, float leafValue) {
  // leafValue is from the perspective of the player to move at the leaf. The edge
  // into the leaf was chosen by the other player: sign starts negative.
  float sign = -1.0f;
  for (size_t i = path.size(); i-- > 0;) {
    Edge* e = path[i];
    e->N += 1;
    e->W += sign * leafValue;
    if (e->virtualLoss > 0) e->virtualLoss -= 1;
    nodes[i]->visitCount += 1;
    sign = -sign;
  }
  if (leaf.state == NodeState::Terminal) {
    leaf.visitCount += 1;
    ++terminalSimulationsThisMove_;
  }
  ++simulationsThisMove_;
}

void SearchTree::addRootNoise() {
  if (noiseApplied_ || !params_.addRootNoise || root_->state != NodeState::Expanded) return;
  const int k = static_cast<int>(root_->edges.size());
  if (k == 0) return;
  std::vector<float> noise(k);
  rng_.dirichlet(params_.dirichletAlpha, noise.data(), k);
  const float eps = params_.dirichletEpsilon;
  for (int i = 0; i < k; ++i) root_->edges[i].P = (1.0f - eps) * root_->edges[i].P + eps * noise[i];
  noiseApplied_ = true;
}

// ---------------------------------------------------------------------------
// Collect / commit protocol

bool SearchTree::budgetExhausted() const {
  if (params_.budgetIncludesInherited) return inheritedVisits_ + simulationsThisMove_ >= static_cast<uint32_t>(params_.simulations);
  return simulationsThisMove_ >= params_.simulations;
}

CollectResult SearchTree::collectLeaf(PendingLeaf& leaf) {
  leaf.path.clear();
  leaf.nodes.clear();
  leaf.hashes.clear();
  leaf.board = rootBoard_;
  Node* node = root_.get();
  for (;;) {
    if (node->state == NodeState::Terminal) {
      backup(leaf.path, leaf.nodes, *node, node->terminalValue);
      return CollectResult::Completed;
    }
    if (node->state == NodeState::Pending) {
      for (Edge* e : leaf.path)
        if (e->virtualLoss > 0) e->virtualLoss -= 1;
      leaf.path.clear();
      leaf.nodes.clear();
      return CollectResult::Collision;
    }
    if (node->state == NodeState::Unexpanded) {
      if (leaf.board.gameOver()) {
        makeTerminal(*node, leaf.board);
        backup(leaf.path, leaf.nodes, *node, node->terminalValue);
        return CollectResult::Completed;
      }
      node->state = NodeState::Pending;
      node->toMove = leaf.board.toMove();
      leaf.node = node;
      leaf.symmetry = params_.searchSymmetry ? static_cast<int>(rng_.uniformInt(kNumSymmetries)) : 0;
      leaf.planes.resize(featureSize(n_));
      if (leaf.symmetry == 0) {
        encodeFeatures(leaf.board, leaf.planes.data());
      } else {
        std::vector<uint8_t> raw(featureSize(n_));
        encodeFeatures(leaf.board, raw.data());
        transformPlanes(raw.data(), leaf.planes.data(), n_, kNumPlanes, leaf.symmetry);
      }
      ++pendingCount_;
      return CollectResult::Pending;
    }
    // Expanded: select and descend.
    const int idx = selectEdge(*node);
    if (idx < 0) throw std::logic_error("expanded node without edges");
    Edge& e = node->edges[idx];
    e.virtualLoss += 1;
    leaf.path.push_back(&e);
    leaf.nodes.push_back(node);
    leaf.board.play(e.move);
    leaf.hashes.push_back(leaf.board.hash());
    if (!e.child) e.child = std::make_unique<Node>();
    node = e.child.get();
  }
}

void SearchTree::commit(PendingLeaf& leaf, const NNOutput& out) {
  Node& node = *leaf.node;
  if (node.state != NodeState::Pending) throw std::logic_error("commit on a node that is not pending");
  HashHistory hh(*hist_, leaf.hashes.data(), static_cast<int>(leaf.hashes.size()));
  expandInto(node, leaf.board, hh, out, leaf.symmetry);
  --pendingCount_;
  if (&node == root_.get()) {
    addRootNoise();  // root expansion: not a simulation
    return;
  }
  backup(leaf.path, leaf.nodes, node, node.value());
}

void SearchTree::abort(PendingLeaf& leaf) {
  Node& node = *leaf.node;
  if (node.state != NodeState::Pending) return;
  node.state = NodeState::Unexpanded;
  for (Edge* e : leaf.path)
    if (e->virtualLoss > 0) e->virtualLoss -= 1;
  --pendingCount_;
  leaf.path.clear();
  leaf.nodes.clear();
}

void SearchTree::prepareRoot() {
  if (root_->state == NodeState::Expanded) addRootNoise();
}

void SearchTree::runSequential(NNEvaluator& ev) {
  if (rootBoard_.gameOver()) return;
  PendingLeaf leaf;
  std::vector<NNInput> in(1);
  std::vector<NNOutput> out;
  // Root expansion (does not count against the budget), then noise. The batched
  // driver (selfplay/batch_runner.cpp) issues exactly this sequence of calls per game.
  prepareRoot();
  if (root_->state == NodeState::Unexpanded) {
    CollectResult r = collectLeaf(leaf);
    if (r == CollectResult::Pending) {
      in[0] = leaf.input();
      try {
        ev.evaluate(in, out);
      } catch (...) {
        abort(leaf);  // root back to Unexpanded; a retry starts cleanly
        throw;
      }
      commit(leaf, out[0]);
    }
  }
  while (!budgetExhausted()) {
    CollectResult r = collectLeaf(leaf);
    if (r == CollectResult::Pending) {
      in[0] = leaf.input();
      try {
        ev.evaluate(in, out);
      } catch (...) {
        abort(leaf);
        throw;
      }
      commit(leaf, out[0]);
    } else if (r == CollectResult::Collision) {
      throw std::logic_error("collision in sequential search");
    }
  }
}

// ---------------------------------------------------------------------------
// Results

float SearchTree::rootMaxQ() const {
  float best = -2.0f;
  for (const Edge& e : root_->edges)
    if (e.N > 0) best = std::max(best, e.Q());
  return best;
}

bool SearchTree::shouldResign() const {
  if (params_.resignThreshold <= -1.0f || root_->state != NodeState::Expanded) return false;
  const float maxQ = rootMaxQ();
  if (maxQ <= -2.0f) return false;  // nothing visited yet
  return rootValue() < params_.resignThreshold && maxQ < params_.resignThreshold;
}

void SearchTree::rootVisits(std::vector<std::pair<Move, uint32_t>>& out) const {
  out.clear();
  for (const Edge& e : root_->edges)
    if (e.N > 0) out.emplace_back(e.move, e.N);
}

void SearchTree::policyTarget(std::vector<float>& out) const {
  const int nn = n_ * n_;
  out.assign(nn + 1, 0.0f);
  const uint32_t total = rootTotalVisits();
  if (total == 0) return;
  for (const Edge& e : root_->edges) out[e.move == kPass ? nn : e.move] = static_cast<float>(e.N) / static_cast<float>(total);
}

Move SearchTree::selectMove(int moveNumber) {
  if (root_->state != NodeState::Expanded || root_->edges.empty()) return kPass;
  const bool sample = moveNumber < params_.temperatureMoves;
  if (sample) {
    const uint32_t total = rootTotalVisits();
    if (total > 0) {
      uint32_t r = rng_.uniformInt(total);
      for (const Edge& e : root_->edges) {
        if (r < e.N) return e.move;
        r -= e.N;
      }
    }
  }
  // argmax N, ties broken by prior, then lower index.
  const Edge* best = nullptr;
  for (const Edge& e : root_->edges)
    if (!best || e.N > best->N || (e.N == best->N && e.P > best->P)) best = &e;
  return best->move;
}

std::string SearchTree::analyzeString(int topK) const {
  std::vector<const Edge*> es;
  for (const Edge& e : root_->edges) es.push_back(&e);
  std::sort(es.begin(), es.end(), [](const Edge* a, const Edge* b) { return a->N != b->N ? a->N > b->N : a->P > b->P; });
  std::ostringstream os;
  os << "root v=" << rootValue() << " visits=" << rootTotalVisits() << " sims=" << simulationsThisMove_ << "\n";
  for (int i = 0; i < std::min<int>(topK, static_cast<int>(es.size())); ++i) {
    const Edge* e = es[i];
    os << (e->move == kPass ? std::string("pass") : std::to_string(rowOfPoint(e->move, n_)) + "," + std::to_string(colOfPoint(e->move, n_)))
       << " N=" << e->N << " Q=" << e->Q() << " P=" << e->P << "\n";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// Debug helpers

const Node* SearchTree::forceLine(const std::vector<Move>& moves, NNEvaluator& ev) {
  if (pendingCount_ != 0) throw std::logic_error("forceLine with pending evaluations");
  Board board = rootBoard_;
  std::vector<uint64_t> hashes;
  Node* node = root_.get();
  std::vector<NNInput> in(1);
  std::vector<NNOutput> out;
  auto expandHere = [&](Node& nd) {
    if (nd.state != NodeState::Unexpanded) return;
    if (board.gameOver()) {
      makeTerminal(nd, board);
      return;
    }
    std::vector<uint8_t> planes(featureSize(n_));
    encodeFeatures(board, planes.data());
    in[0] = NNInput{planes.data()};
    ev.evaluate(in, out);
    HashHistory hh(*hist_, hashes.data(), static_cast<int>(hashes.size()));
    expandInto(nd, board, hh, out[0], 0);
  };
  expandHere(*node);
  for (Move m : moves) {
    if (node->state != NodeState::Expanded) return nullptr;
    Edge* found = nullptr;
    for (Edge& e : node->edges)
      if (e.move == m) found = &e;
    if (!found) return nullptr;
    if (!found->child) found->child = std::make_unique<Node>();
    board.play(m);
    hashes.push_back(board.hash());
    node = found->child.get();
    expandHere(*node);
  }
  return node;
}

std::string SearchTree::checkInvariantsRec(const Node& node) const {
  if (node.state == NodeState::Pending) return "pending node found";
  if (node.state == NodeState::Expanded) {
    if (node.visitCount != 1 + node.totalEdgeVisits())
      return "visitCount " + std::to_string(node.visitCount) + " != 1 + sumN " + std::to_string(node.totalEdgeVisits());
    for (const Edge& e : node.edges) {
      if (e.virtualLoss != 0) return "virtual loss left on an edge";
      if (e.child) {
        std::string s = checkInvariantsRec(*e.child);
        if (!s.empty()) return s;
        // The evaluation that expanded the child is both the child's first visit and
        // the edge's first visit, so child.visitCount == edge.N (Expanded or Terminal).
        if ((e.child->state == NodeState::Expanded || e.child->state == NodeState::Terminal) && e.child->visitCount != e.N)
          return "child visitCount " + std::to_string(e.child->visitCount) + " != edge N " + std::to_string(e.N);
        if (e.child->state == NodeState::Unexpanded && e.N != 0) return "unexpanded child with visits";
      } else if (e.N != 0) {
        return "edge with visits but no child";
      }
    }
  }
  return "";
}

std::string SearchTree::checkInvariants() const {
  if (pendingCount_ != 0) return "pendingCount != 0";
  return checkInvariantsRec(*root_);
}

}  // namespace mango
