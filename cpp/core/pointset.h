// Fixed-size bitset over unbordered board points (up to 19*19 = 361 bits).
#pragma once

#include <bit>
#include <cstdint>

#include "core/types.h"

namespace mango {

struct PointSet {
  static constexpr int kWords = (kMaxPoints + 63) / 64;  // 6
  uint64_t w[kWords] = {};

  void set(int p) { w[p >> 6] |= (uint64_t{1} << (p & 63)); }
  void clear(int p) { w[p >> 6] &= ~(uint64_t{1} << (p & 63)); }
  bool test(int p) const { return (w[p >> 6] >> (p & 63)) & 1; }
  void reset() {
    for (auto& x : w) x = 0;
  }
  int count() const {
    int n = 0;
    for (auto x : w) n += std::popcount(x);
    return n;
  }
  bool any() const {
    for (auto x : w)
      if (x) return true;
    return false;
  }
  void orWith(const PointSet& o) {
    for (int i = 0; i < kWords; ++i) w[i] |= o.w[i];
  }
  // Index of the lowest set bit, or -1.
  int first() const {
    for (int i = 0; i < kWords; ++i)
      if (w[i]) return i * 64 + std::countr_zero(w[i]);
    return -1;
  }
  bool operator==(const PointSet& o) const {
    for (int i = 0; i < kWords; ++i)
      if (w[i] != o.w[i]) return false;
    return true;
  }
  bool operator!=(const PointSet& o) const { return !(*this == o); }
};

}  // namespace mango
