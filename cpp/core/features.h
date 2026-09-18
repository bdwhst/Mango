// AGZ feature encoding (features_v1): 17 planes of n*n uint8 (0/1).
//   planes 0..7  : stones of the player to move at t, t-1, ..., t-7
//   planes 8..15 : stones of the opponent at t, t-1, ..., t-7
//   plane 16     : all ones if black is to move, else all zeros
#pragma once

#include <cstdint>

#include "core/board.h"

namespace mango {

inline int featureSize(int n) { return kNumPlanes * n * n; }

// out must hold featureSize(board.size()) bytes.
void encodeFeatures(const Board& board, uint8_t* out);

}  // namespace mango
