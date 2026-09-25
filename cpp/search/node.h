// Search tree nodes and edges (docs/DESIGN.md section 5.4.2).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/types.h"

namespace mango {

enum class NodeState : uint8_t { Unexpanded, Pending, Expanded, Terminal };

struct Node;

struct Edge {
  Move move = kNoMove;
  uint16_t virtualLoss = 0;
  float P = 0.0f;
  uint32_t N = 0;
  float W = 0.0f;  // sum of backed-up values, perspective of the player choosing this edge
  // Exact value of this move when it ends the game (second consecutive pass, or the
  // move cap), perspective of the chooser; kNoTerminal otherwise (DESIGN 5.4.10).
  float terminalValue = kNoTerminal;
  std::unique_ptr<Node> child;

  static constexpr float kNoTerminal = -2.0f;
  float Q() const { return N > 0 ? W / static_cast<float>(N) : 0.0f; }
  bool endsGame() const { return terminalValue > kNoTerminal; }
};

struct Node {
  NodeState state = NodeState::Unexpanded;
  Color toMove = Color::Black;
  float nnValue = 0.0f;        // v(s) from the network, perspective of toMove (valid once Expanded)
  float terminalValue = 0.0f;  // valid iff Terminal, perspective of toMove
  // Best exact value among the game-ending moves at this node (perspective of toMove),
  // Edge::kNoTerminal if none: toMove can guarantee at least this much (DESIGN 5.4.10).
  float terminalMoveValue = Edge::kNoTerminal;
  uint32_t visitCount = 0;     // Expanded: 1 + sum of edge N; Terminal: times reached
  std::vector<Edge> edges;     // legal moves only

  // The value backed up for an expanded node: the network value, raised to the exact
  // value of a game-ending move when that is better (a minimax lower bound).
  float value() const { return terminalMoveValue > nnValue ? terminalMoveValue : nnValue; }

  Node() { ++liveCount(); }
  ~Node() { --liveCount(); }
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  uint32_t totalEdgeVisits() const {
    uint32_t s = 0;
    for (const Edge& e : edges) s += e.N;
    return s;
  }

  // Number of Node objects currently alive in the process (diagnostics: peak memory).
  static std::atomic<int64_t>& liveCount() {
    static std::atomic<int64_t> c{0};
    return c;
  }
};

}  // namespace mango
