#include "core/features.h"

namespace mango {

void encodeFeatures(const Board& board, uint8_t* out) {
  const int n = board.size();
  const int nn = n * n;
  const int me = colorIndex(board.toMove());
  const int opp = 1 - me;
  for (int k = 0; k < kHistoryLen; ++k) {
    const Snapshot& s = board.snapshot(k);
    uint8_t* mine = out + k * nn;
    uint8_t* theirs = out + (kHistoryLen + k) * nn;
    for (int p = 0; p < nn; ++p) {
      mine[p] = s.stones[me].test(p) ? 1 : 0;
      theirs[p] = s.stones[opp].test(p) ? 1 : 0;
    }
  }
  const uint8_t colorPlane = board.toMove() == Color::Black ? 1 : 0;
  uint8_t* last = out + 2 * kHistoryLen * nn;
  for (int p = 0; p < nn; ++p) last[p] = colorPlane;
}

}  // namespace mango
