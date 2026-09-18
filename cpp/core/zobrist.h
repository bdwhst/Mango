// Deterministic Zobrist keys: identical on every platform and run, so that
// hashes stored in files / logs are comparable.
#pragma once

#include <cstdint>

#include "core/types.h"

namespace mango {

inline uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

struct ZobristTable {
  uint64_t key[2][kMaxCells];  // [colorIndex][bordered loc]
  ZobristTable() {
    uint64_t s = 0x4D616E676F5A6F62ull;  // "MangoZob"
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < kMaxCells; ++i) key[c][i] = splitmix64(s);
  }
};

inline const ZobristTable& zobrist() {
  static const ZobristTable table;
  return table;
}

inline uint64_t zobristKey(Color c, int loc) { return zobrist().key[colorIndex(c)][loc]; }

}  // namespace mango
