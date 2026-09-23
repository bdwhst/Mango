// PUCT search for one game (docs/DESIGN.md sections 5.4.1-5.4.8).
//
// The tree exposes a collect / commit protocol so that one driver can batch leaf
// evaluations across many games (and, later, K leaves within a game). Sequential
// search is the special case K = 1 with a synchronous evaluator.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/board.h"
#include "core/history.h"
#include "core/random.h"
#include "nn/evaluator.h"
#include "search/node.h"
#include "search/search_params.h"

namespace mango {

// One outstanding leaf evaluation.
struct PendingLeaf {
  Node* node = nullptr;
  std::vector<Edge*> path;        // root -> leaf edges (empty for the root itself)
  std::vector<Node*> nodes;       // root -> parent of leaf (nodes.size() == path.size())
  std::vector<uint64_t> hashes;   // position hashes along the path (after each edge)
  Board board;                    // position at the leaf
  int symmetry = 0;               // symmetry applied to the planes (policy is mapped back on commit)
  std::vector<uint8_t> planes;    // encoded (and symmetrised) features
  NNInput input() const { return NNInput{planes.data()}; }
  PendingLeaf() : board(2, 0.0f) {}
};

enum class CollectResult { Pending, Completed, Collision };

class SearchTree {
 public:
  SearchTree(const SearchParams& params, int boardSize, uint64_t seed);

  // Starts a fresh tree at this position (history is owned by the caller and must
  // outlive the tree / be replaced via advance()).
  void newGame(const Board& board, const GameHistory& hist);
  // The game advanced by `move` to `boardAfter`. With tree reuse the corresponding
  // subtree becomes the root; otherwise a fresh root is created.
  void advance(Move move, const Board& boardAfter, const GameHistory& hist);

  // --- Sequential search (K = 1, synchronous evaluator) ------------------------
  void runSequential(NNEvaluator& ev);

  // --- Batched protocol ---------------------------------------------------------
  // Call once at the start of every move before collecting leaves: a reused
  // (already expanded) root gets its root noise here, exactly as runSequential does.
  // An unexpanded root is expanded by the first collectLeaf/commit (not a simulation).
  void prepareRoot();
  bool rootExpanded() const { return root_->state == NodeState::Expanded; }
  bool gameOverAtRoot() const { return rootBoard_.gameOver(); }
  int pendingCount() const { return pendingCount_; }
  // Descends from the root with virtual loss. Pending: `leaf` holds a request that
  // must be committed or aborted. Completed: a terminal was backed up (counts as a
  // simulation). Collision: a Pending node was reached; reservations were removed.
  CollectResult collectLeaf(PendingLeaf& leaf);
  void commit(PendingLeaf& leaf, const NNOutput& out);
  void abort(PendingLeaf& leaf);  // evaluation failed: node back to Unexpanded, reservations removed
  bool budgetExhausted() const;

  // --- Results ------------------------------------------------------------------
  const Node& root() const { return *root_; }
  Node& rootMutable() { return *root_; }
  int simulationsThisMove() const { return simulationsThisMove_; }
  // Simulations this move that ended at a terminal node (no network evaluation).
  int terminalSimulationsThisMove() const { return terminalSimulationsThisMove_; }
  uint32_t rootTotalVisits() const { return root_->totalEdgeVisits(); }
  uint32_t inheritedVisits() const { return inheritedVisits_; }
  float rootValue() const { return root_->nnValue; }         // v(s0), root player's perspective
  float rootMaxQ() const;                                      // max over visited edges, same perspective
  bool shouldResign() const;                                   // both rootValue and rootMaxQ below the threshold
  // Sparse root visit counts (move, N) for every edge with N > 0.
  void rootVisits(std::vector<std::pair<Move, uint32_t>>& out) const;
  // Normalised visit distribution over n*n+1 actions (pass = index n*n).
  void policyTarget(std::vector<float>& out) const;
  // Move choice with the temperature rule (tau = 1 below temperatureMoves, argmax after).
  Move selectMove(int moveNumber);
  std::string analyzeString(int topK = 10) const;

  int boardSize() const { return n_; }
  const SearchParams& params() const { return params_; }
  void setSimulations(int s) { params_.simulations = s; }
  Rng& rng() { return rng_; }

  // Debug/test helper: forces expansion along a line of moves (evaluating with `ev`
  // as needed) and returns the node at its end, or nullptr if a move is not a legal
  // edge along the way. Expansions are not backed up, so checkInvariants() will
  // report the forced nodes afterwards; never use this in real search.
  const Node* forceLine(const std::vector<Move>& moves, NNEvaluator& ev);
  // Debug/test helper: checks the tree invariants (no Pending, no virtual loss,
  // visitCount == 1 + sum N for expanded nodes). Returns an empty string if fine.
  std::string checkInvariants() const;

  // The index of the edge PUCT would select at this node (public for tests).
  int selectEdge(const Node& node) const;

 private:
  void expandInto(Node& node, const Board& board, const HashHistory& hist, const NNOutput& out, int symmetry);
  void makeTerminal(Node& node, const Board& board);
  void backup(const std::vector<Edge*>& path, const std::vector<Node*>& nodes, Node& leaf, float leafValue);
  void addRootNoise();
  void resetRoot(const Board& board, const GameHistory& hist);
  std::string checkInvariantsRec(const Node& node) const;

  SearchParams params_;
  int n_;
  Rng rng_;
  std::unique_ptr<Node> root_;
  Board rootBoard_;
  const GameHistory* hist_ = nullptr;
  int simulationsThisMove_ = 0;
  int terminalSimulationsThisMove_ = 0;
  uint32_t inheritedVisits_ = 0;
  bool noiseApplied_ = false;
  int pendingCount_ = 0;
  std::vector<float> scratchPolicy_;
};

}  // namespace mango
