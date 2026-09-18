// Seedable RNG (xoshiro256**) and seed derivation helpers.
#pragma once

#include <cmath>
#include <cstdint>

#include "core/zobrist.h"

namespace mango {

class Rng {
 public:
  explicit Rng(uint64_t seed) { reseed(seed); }

  void reseed(uint64_t seed) {
    uint64_t sm = seed;
    for (auto& x : s_) x = splitmix64(sm);
  }

  uint64_t next() {
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform integer in [0, n). n must be > 0.
  uint32_t uniformInt(uint32_t n) {
    // Lemire's nearly-divisionless method.
    uint64_t m = static_cast<uint64_t>(static_cast<uint32_t>(next())) * n;
    uint32_t l = static_cast<uint32_t>(m);
    if (l < n) {
      uint32_t t = (0u - n) % n;
      while (l < t) {
        m = static_cast<uint64_t>(static_cast<uint32_t>(next())) * n;
        l = static_cast<uint32_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 32);
  }

  // Uniform real in [0, 1).
  double uniformReal() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  // Standard normal (Box-Muller).
  double normal() {
    double u1 = uniformReal();
    double u2 = uniformReal();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }

  // Gamma(alpha, 1) via Marsaglia-Tsang (with the alpha < 1 boost).
  double gamma(double alpha) {
    if (alpha < 1.0) {
      double u = uniformReal();
      if (u < 1e-300) u = 1e-300;
      return gamma(alpha + 1.0) * std::pow(u, 1.0 / alpha);
    }
    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      double x, v;
      do {
        x = normal();
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      const double u = uniformReal();
      if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
    }
  }

  // Symmetric Dirichlet(alpha) sample of length n, written to out (sums to 1).
  void dirichlet(double alpha, float* out, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      double g = gamma(alpha);
      out[i] = static_cast<float>(g);
      sum += g;
    }
    if (sum <= 0.0) {
      for (int i = 0; i < n; ++i) out[i] = 1.0f / static_cast<float>(n);
      return;
    }
    for (int i = 0; i < n; ++i) out[i] = static_cast<float>(out[i] / sum);
  }

 private:
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t s_[4];
};

// Deterministic seed derivation: hash of (a, b, c...).
inline uint64_t deriveSeed(uint64_t a, uint64_t b) {
  uint64_t s = a ^ 0x9E3779B97F4A7C15ull;
  uint64_t h = splitmix64(s);
  s ^= b + 0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
  return splitmix64(s);
}
inline uint64_t deriveSeed(uint64_t a, uint64_t b, uint64_t c) { return deriveSeed(deriveSeed(a, b), c); }

}  // namespace mango
