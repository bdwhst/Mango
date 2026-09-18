#include "core/board.h"
#include "core/features.h"
#include "core/symmetry.h"
#include "doctest/doctest.h"
#include "tests/test_util.h"

using namespace mango;
using namespace mango::test;

namespace {
bool snapshotEquals(const Snapshot& s, const std::vector<Color>& pos, int n) {
  for (int p = 0; p < n * n; ++p) {
    bool b = pos[p] == Color::Black, w = pos[p] == Color::White;
    if (s.stones[0].test(p) != b || s.stones[1].test(p) != w) return false;
  }
  return true;
}
}  // namespace

TEST_CASE("snapshots track the last 8 positions including captures and passes") {
  for (int n : {5, 9}) {
    for (uint64_t seed = 11; seed < 16; ++seed) {
      randomGame(n, 7.5f, seed, 0.05, [&](const Board& b, const RefBoard& ref, const GameHistory&) {
        const int have = static_cast<int>(ref.positions.size());
        CHECK(b.snapshotCount() == std::min(have, kHistoryLen));
        for (int k = 0; k < kHistoryLen; ++k) {
          const Snapshot& s = b.snapshot(k);
          int idx = have - 1 - k;
          if (idx < 0) {
            CHECK_FALSE(s.stones[0].any());
            CHECK_FALSE(s.stones[1].any());
          } else {
            CHECK(snapshotEquals(s, ref.positions[idx], n));
          }
        }
      });
    }
  }
}

TEST_CASE("a pass duplicates the current snapshot") {
  Board b(5, 7.5f);
  b.play(static_cast<Move>(12));
  b.play(kPass);
  CHECK(b.snapshot(0) == b.snapshot(1));
  CHECK_FALSE(b.snapshot(0) == b.snapshot(2));
  CHECK(b.snapshotCount() == 3);
}

TEST_CASE("features match the reference encoding") {
  for (int n : {5, 7, 9}) {
    for (uint64_t seed = 21; seed < 25; ++seed) {
      std::vector<uint8_t> planes(featureSize(n));
      randomGame(n, 7.5f, seed, 0.03, [&](const Board& b, const RefBoard& ref, const GameHistory&) {
        encodeFeatures(b, planes.data());
        REQUIRE(planes == ref.features());
      });
    }
  }
}

TEST_CASE("applySymmetry transforms cells and all snapshots consistently") {
  const int n = 7;
  std::vector<uint8_t> planes(featureSize(n)), transformed(featureSize(n)), planesSym(featureSize(n));
  for (uint64_t seed = 31; seed < 34; ++seed) {
    randomGame(n, 7.5f, seed, 0.03, [&](const Board& b, const RefBoard&, const GameHistory&) {
      encodeFeatures(b, planes.data());
      for (int sym = 0; sym < kNumSymmetries; ++sym) {
        Board s = b.applySymmetry(sym);
        REQUIRE(s.applySymmetry(inverseSymmetry(sym)) == b);
        // Cells transformed.
        for (int p = 0; p < n * n; ++p) REQUIRE(s.atPoint(transformPoint(p, n, sym)) == b.atPoint(p));
        // Chains rebuilt correctly: liberties are symmetry-invariant.
        for (int p = 0; p < n * n; ++p) REQUIRE(s.liberties(transformPoint(p, n, sym)) == b.liberties(p));
        // Planes of the symmetric board == symmetric planes of the board.
        encodeFeatures(s, planesSym.data());
        transformPlanes(planes.data(), transformed.data(), n, kNumPlanes, sym);
        REQUIRE(planesSym == transformed);
      }
    });
  }
}
