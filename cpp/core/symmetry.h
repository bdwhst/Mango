// The 8 dihedral symmetries of the board and how they act on points, moves,
// feature planes and policy vectors.
//
// Encoding of sym in [0, 8): bit 0 = transpose, bit 1 = flip rows (top<->bottom),
// bit 2 = flip columns (left<->right). Applied in that order: transpose first.
#pragma once

#include <cstdint>

#include "core/types.h"

namespace mango {

constexpr int kNumSymmetries = 8;

int transformPoint(int p, int n, int sym);
int inverseSymmetry(int sym);
Move transformMove(Move m, int n, int sym);  // pass is unchanged

// out[plane][T(p)] = in[plane][p] for every plane. in/out must not alias.
void transformPlanes(const uint8_t* in, uint8_t* out, int n, int numPlanes, int sym);

// Policy vector of length n*n+1 (last entry = pass, unchanged). in/out must not alias.
void transformPolicy(const float* in, float* out, int n, int sym);

}  // namespace mango
