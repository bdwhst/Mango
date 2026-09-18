// Neural-network evaluator interface (docs/DESIGN.md section 5.3) and the
// deterministic FakeEvaluator used by search tests. This header has no LibTorch
// dependency.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/types.h"

namespace mango {

struct NNInput {
  const uint8_t* planes;  // 17*n*n, owned by the caller, valid until evaluate() returns
};

struct NNOutput {
  std::vector<float> policy;  // n*n+1 probabilities (fp32 softmax over ALL actions, before legal masking)
  float value = 0.0f;         // in [-1, 1], perspective of the player to move
};

class NNEvaluator {
 public:
  virtual ~NNEvaluator() = default;
  virtual int boardSize() const = 0;
  virtual const std::string& modelId() const = 0;
  // Synchronous batch evaluation. out is resized to in.size().
  virtual void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) = 0;
};

// Deterministic evaluator for tests. The callback receives the planes and must fill
// `policy` (n*n+1 entries, need not be normalised: it is normalised here) and `value`.
class FakeEvaluator : public NNEvaluator {
 public:
  using Fn = std::function<void(const uint8_t* planes, float* policy, float* value)>;

  // Uniform policy, constant value.
  FakeEvaluator(int n, float value = 0.0f) : n_(n), id_("fake"), fn_(nullptr), constValue_(value) {}
  FakeEvaluator(int n, Fn fn) : n_(n), id_("fake"), fn_(std::move(fn)), constValue_(0.0f) {}

  int boardSize() const override { return n_; }
  const std::string& modelId() const override { return id_; }
  int calls() const { return calls_; }
  int positions() const { return positions_; }

  void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override {
    ++calls_;
    positions_ += static_cast<int>(in.size());
    const int na = n_ * n_ + 1;
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      out[i].policy.assign(na, 0.0f);
      if (fn_) {
        fn_(in[i].planes, out[i].policy.data(), &out[i].value);
        double sum = 0.0;
        for (float p : out[i].policy) sum += p;
        if (sum > 0)
          for (float& p : out[i].policy) p = static_cast<float>(p / sum);
        else
          for (float& p : out[i].policy) p = 1.0f / static_cast<float>(na);
      } else {
        for (float& p : out[i].policy) p = 1.0f / static_cast<float>(na);
        out[i].value = constValue_;
      }
    }
  }

 private:
  int n_;
  std::string id_;
  Fn fn_;
  float constValue_;
  int calls_ = 0;
  int positions_ = 0;
};

}  // namespace mango
