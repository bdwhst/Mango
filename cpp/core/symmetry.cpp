#include "core/symmetry.h"

#include <array>
#include <utility>

namespace mango {

namespace {

constexpr int transformPointImpl(int p, int n, int sym) {
  int r = p / n, c = p % n;
  if (sym & 1) std::swap(r, c);
  if (sym & 2) r = n - 1 - r;
  if (sym & 4) c = n - 1 - c;
  return r * n + c;
}

// Inverse table computed at compile time by brute force on a generic 4x4 board; the
// group action does not depend on n. Being a constant, it is trivially thread-safe.
constexpr std::array<int, kNumSymmetries> makeInverseTable() {
  std::array<int, kNumSymmetries> table{};
  for (int sym = 0; sym < kNumSymmetries; ++sym) {
    table[sym] = -1;
    for (int s2 = 0; s2 < kNumSymmetries && table[sym] < 0; ++s2) {
      bool ok = true;
      for (int p = 0; p < 16 && ok; ++p) ok = transformPointImpl(transformPointImpl(p, 4, sym), 4, s2) == p;
      if (ok) table[sym] = s2;
    }
  }
  return table;
}

constexpr std::array<int, kNumSymmetries> kInverse = makeInverseTable();
static_assert(kInverse[0] == 0 && kInverse[3] == 5 && kInverse[5] == 3, "unexpected symmetry inverses");

}  // namespace

int transformPoint(int p, int n, int sym) { return transformPointImpl(p, n, sym); }

int inverseSymmetry(int sym) { return kInverse[sym]; }

Move transformMove(Move m, int n, int sym) {
  if (m < 0) return m;
  return static_cast<Move>(transformPointImpl(m, n, sym));
}

void transformPlanes(const uint8_t* in, uint8_t* out, int n, int numPlanes, int sym) {
  const int nn = n * n;
  for (int plane = 0; plane < numPlanes; ++plane) {
    const uint8_t* src = in + plane * nn;
    uint8_t* dst = out + plane * nn;
    for (int p = 0; p < nn; ++p) dst[transformPointImpl(p, n, sym)] = src[p];
  }
}

void transformPolicy(const float* in, float* out, int n, int sym) {
  const int nn = n * n;
  for (int p = 0; p < nn; ++p) out[transformPointImpl(p, n, sym)] = in[p];
  out[nn] = in[nn];
}

}  // namespace mango
