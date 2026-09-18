// Game history (moves + position hashes) owned by the game, and the light-weight
// view the search passes into legality checks (game history + path hashes).
#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core/types.h"

namespace mango {

struct GameHistory {
  std::vector<Move> moves;
  std::vector<uint64_t> hashes;          // hashes[0] = initial position, hashes[i] = after moves[i-1]
  std::unordered_set<uint64_t> hashSet;  // same content as hashes

  void reset(uint64_t initialHash) {
    moves.clear();
    hashes.clear();
    hashSet.clear();
    hashes.push_back(initialHash);
    hashSet.insert(initialHash);
  }
  void push(Move m, uint64_t hashAfter) {
    moves.push_back(m);
    hashes.push_back(hashAfter);
    hashSet.insert(hashAfter);
  }
  bool contains(uint64_t h) const { return hashSet.count(h) != 0; }
  size_t size() const { return moves.size(); }
};

// Positions that a placement may not recreate: everything in the game so far plus
// the positions along the current search path (small, scanned linearly).
struct HashHistory {
  const GameHistory* game = nullptr;
  const uint64_t* path = nullptr;
  int pathLen = 0;

  HashHistory() = default;
  explicit HashHistory(const GameHistory& g) : game(&g) {}
  HashHistory(const GameHistory& g, const uint64_t* p, int len) : game(&g), path(p), pathLen(len) {}

  bool contains(uint64_t h) const {
    if (game && game->contains(h)) return true;
    for (int i = 0; i < pathLen; ++i)
      if (path[i] == h) return true;
    return false;
  }
};

}  // namespace mango
