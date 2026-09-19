// NN backend parity test (docs/DESIGN.md section 9, "NN backend"). Loads the committed
// fixture model and compares the C++ evaluator on every available device against
// reference outputs computed by PyTorch in fp32.
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "nlohmann/json.hpp"
#include "nn/torch_evaluator.h"

using namespace mango;

#ifndef MANGO_FIXTURE_DIR
#define MANGO_FIXTURE_DIR "cpp/tests/fixtures"
#endif

namespace {

struct Reference {
  int n = 0;
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<std::vector<float>> policy;  // softmax of the reference logits (computed here in double)
  std::vector<float> values;
};

Reference loadReference(const std::string& dir) {
  std::ifstream in(dir + "/reference.json");
  REQUIRE_MESSAGE(in.good(), "missing fixture: run python python/tests/make_fixture.py");
  nlohmann::json j = nlohmann::json::parse(in);
  Reference r;
  r.n = j["board_size"].get<int>();
  for (const auto& row : j["inputs"]) r.inputs.push_back(row.get<std::vector<uint8_t>>());
  for (const auto& row : j["logits"]) {
    std::vector<double> l = row.get<std::vector<double>>();
    double mx = l[0];
    for (double x : l) mx = std::max(mx, x);
    double sum = 0.0;
    for (double x : l) sum += std::exp(x - mx);
    std::vector<float> p;
    for (double x : l) p.push_back(static_cast<float>(std::exp(x - mx) / sum));
    r.policy.push_back(p);
  }
  r.values = j["values"].get<std::vector<float>>();
  return r;
}

// Returns the largest |policy - reference| over all rows and actions.
double checkAgainstReference(TorchEvaluator& ev, const Reference& ref, bool half) {
  std::vector<NNInput> in;
  for (const auto& row : ref.inputs) in.push_back(NNInput{row.data()});
  std::vector<NNOutput> out;
  ev.evaluate(in, out);
  REQUIRE(out.size() == ref.inputs.size());
  const int na = ref.n * ref.n + 1;
  double worst = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    REQUIRE(static_cast<int>(out[i].policy.size()) == na);
    double sum = 0.0, kl = 0.0, maxAbs = 0.0;
    for (int a = 0; a < na; ++a) {
      sum += out[i].policy[a];
      maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(out[i].policy[a] - ref.policy[i][a])));
      const double p = ref.policy[i][a], q = out[i].policy[a];
      if (p > 0) kl += p * std::log(p / std::max(q, 1e-30));
      if (p > 1e-4) CHECK_MESSAGE(q > 0.0f, "prior underflowed to 0 for an action with fp32 prior " << p);
    }
    CHECK(sum == doctest::Approx(1.0).epsilon(1e-5));
    if (half) {
      CHECK(kl < 1e-3);
      CHECK(std::fabs(out[i].value - ref.values[i]) < 1e-2);
    } else {
      CHECK(maxAbs < 1e-4);
      CHECK(std::fabs(out[i].value - ref.values[i]) < 1e-4);
    }
    worst = std::max(worst, std::max(maxAbs, static_cast<double>(std::fabs(out[i].value - ref.values[i]))));
  }
  // Batch of 1 equals the batched result row-wise.
  for (size_t i = 0; i < in.size(); ++i) {
    std::vector<NNInput> one{in[i]};
    std::vector<NNOutput> o1;
    ev.evaluate(one, o1);
    REQUIRE(o1.size() == 1);
    const double tol = half ? 2e-3 : 1e-4;
    for (int a = 0; a < na; ++a) CHECK(std::fabs(o1[0].policy[a] - out[i].policy[a]) < tol);
    CHECK(std::fabs(o1[0].value - out[i].value) < tol);
  }
  return worst;
}

const std::string kFixture = std::string(MANGO_FIXTURE_DIR) + "/model_5x5_v1";

}  // namespace

TEST_CASE("model metadata loads and validates") {
  ModelMeta m = ModelMeta::load(kFixture);
  CHECK(m.boardSize == 5);
  CHECK(m.resBlocks == 2);
  CHECK(m.filters == 16);
  CHECK(m.komi == doctest::Approx(7.5f));
  CHECK(m.moveCap == 50);
  BoardConfig b;
  b.size = 5;
  b.komi = 7.5f;
  CHECK_NOTHROW(m.validate(b));
  b.komi = 6.5f;
  CHECK_THROWS(m.validate(b));
  CHECK_NOTHROW(m.validate(b, /*allowKomiMismatch=*/true));
  b.komi = 7.5f;
  b.size = 9;
  CHECK_THROWS(m.validate(b));
}

TEST_CASE("torch evaluator matches the PyTorch reference on CPU (fp32)") {
  Reference ref = loadReference(kFixture);
  TorchEvaluator::Options o;
  o.device = "cpu";
  TorchEvaluator ev(kFixture, o);
  CHECK(ev.boardSize() == 5);
  CHECK_FALSE(ev.isHalf());
  checkAgainstReference(ev, ref, /*half=*/false);
}

TEST_CASE("torch evaluator matches the PyTorch reference on CUDA (fp32 and fp16)") {
  if (!TorchEvaluator::cudaAvailable()) {
    MESSAGE("CUDA not available: skipped");
    return;
  }
  Reference ref = loadReference(kFixture);
  // TF32 is what LibTorch would use for fp32 convolutions by default. Measure its error
  // first (loose bound only: it is an Ampere+ hardware property, not a contract) so the
  // strict fp32 run below is known to be exercised with TF32 off, not passing by luck.
  double tf32Err = 0.0;
  {
    TorchEvaluator::Options o;
    o.device = "cuda";
    o.fp16 = false;
    o.allowTf32 = true;
    TorchEvaluator ev(kFixture, o);
    std::vector<NNInput> in;
    for (const auto& row : ref.inputs) in.push_back(NNInput{row.data()});
    std::vector<NNOutput> out;
    ev.evaluate(in, out);
    for (size_t i = 0; i < out.size(); ++i) {
      for (size_t a = 0; a < out[i].policy.size(); ++a)
        tf32Err = std::max(tf32Err, static_cast<double>(std::fabs(out[i].policy[a] - ref.policy[i][a])));
      tf32Err = std::max(tf32Err, static_cast<double>(std::fabs(out[i].value - ref.values[i])));
    }
    CHECK(tf32Err < 1e-2);
  }
  {
    TorchEvaluator::Options o;
    o.device = "cuda";
    o.fp16 = false;
    TorchEvaluator ev(kFixture, o);  // allowTf32 defaults to false
    CHECK_FALSE(ev.isHalf());
    const double err = checkAgainstReference(ev, ref, false);
    MESSAGE("CUDA max abs error vs PyTorch fp32 reference: fp32=" << err << " tf32=" << tf32Err);
  }
  {
    TorchEvaluator::Options o;
    o.device = "cuda";
    o.fp16 = true;
    TorchEvaluator ev(kFixture, o);
    CHECK(ev.isHalf());
    checkAgainstReference(ev, ref, true);
  }
}

TEST_CASE("torch evaluator matches the PyTorch reference on MPS (fp32)") {
  if (!TorchEvaluator::mpsAvailable()) {
    MESSAGE("MPS not available: skipped");
    return;
  }
  Reference ref = loadReference(kFixture);
  TorchEvaluator::Options o;
  o.device = "mps";
  TorchEvaluator ev(kFixture, o);
  CHECK_FALSE(ev.isHalf());
  checkAgainstReference(ev, ref, false);
}

TEST_CASE("auto device selection picks an available device") {
  TorchEvaluator::Options o;
  TorchEvaluator ev(kFixture, o);
  std::string d = ev.deviceName();
  CHECK((d.rfind("cuda", 0) == 0 || d.rfind("mps", 0) == 0 || d == "cpu"));
  MESSAGE("auto device: " << d);
}
