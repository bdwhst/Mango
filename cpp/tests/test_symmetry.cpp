#include <vector>

#include "core/symmetry.h"
#include "doctest/doctest.h"

using namespace mango;

TEST_CASE("symmetries are bijections with correct inverses") {
  for (int n : {2, 3, 5, 9, 19}) {
    for (int sym = 0; sym < kNumSymmetries; ++sym) {
      std::vector<int> hit(n * n, 0);
      for (int p = 0; p < n * n; ++p) {
        int q = transformPoint(p, n, sym);
        REQUIRE(q >= 0);
        REQUIRE(q < n * n);
        hit[q]++;
        REQUIRE(transformPoint(q, n, inverseSymmetry(sym)) == p);
      }
      for (int q = 0; q < n * n; ++q) REQUIRE(hit[q] == 1);
    }
    CHECK(inverseSymmetry(0) == 0);
    CHECK(transformMove(kPass, n, 5) == kPass);
  }
  // Explicit table for this encoding (transpose, then flip rows, then flip columns):
  // the flips are involutions, transpose+flipRows inverts to transpose+flipCols.
  const int expected[8] = {0, 1, 2, 5, 4, 3, 6, 7};
  for (int s = 0; s < 8; ++s) CHECK(inverseSymmetry(s) == expected[s]);
}

TEST_CASE("the 8 symmetries are distinct") {
  const int n = 4;
  for (int a = 0; a < kNumSymmetries; ++a)
    for (int b = a + 1; b < kNumSymmetries; ++b) {
      bool same = true;
      for (int p = 0; p < n * n && same; ++p) same = transformPoint(p, n, a) == transformPoint(p, n, b);
      CHECK_FALSE(same);
    }
}

TEST_CASE("policy transform moves a peak and keeps pass in place") {
  const int n = 5;
  std::vector<float> pol(n * n + 1, 0.0f), out(n * n + 1), back(n * n + 1);
  pol[pointOf(1, 2, n)] = 0.7f;
  pol[n * n] = 0.3f;
  for (int sym = 0; sym < kNumSymmetries; ++sym) {
    transformPolicy(pol.data(), out.data(), n, sym);
    CHECK(out[transformPoint(pointOf(1, 2, n), n, sym)] == doctest::Approx(0.7f));
    CHECK(out[n * n] == doctest::Approx(0.3f));
    transformPolicy(out.data(), back.data(), n, inverseSymmetry(sym));
    CHECK(back == pol);
  }
  // Concrete cases: transpose (1,2)->(2,1); flip rows (1,2)->(3,2); flip cols (1,2)->(1,2)... (n-1-2=2).
  CHECK(transformPoint(pointOf(1, 2, n), n, 1) == pointOf(2, 1, n));
  CHECK(transformPoint(pointOf(1, 2, n), n, 2) == pointOf(3, 2, n));
  CHECK(transformPoint(pointOf(1, 2, n), n, 4) == pointOf(1, 2, n));
  CHECK(transformPoint(pointOf(1, 0, n), n, 4) == pointOf(1, 4, n));
}
