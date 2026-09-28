// NN backend parity test (docs/DESIGN.md section 9, "NN backend"). Loads the committed
// fixture model and compares the C++ evaluator on every available device against
// reference outputs computed by PyTorch in fp32.
#include <cmath>
#include <cstdlib>
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

// ---------------------------------------------------------------------------------
// CUDA fast path (DESIGN 5.3, M4a step 2): channels-last parameters and input, one CUDA
// graph per batch bucket. Every switch combination keeps the M2 parity; the fast path
// agrees with the plain path; static buffers do not leak between calls.
#include "core/random.h"

namespace {

std::vector<std::vector<uint8_t>> randomPlanes(int count, int n, uint64_t seed) {
  Rng rng(seed);
  std::vector<std::vector<uint8_t>> rows(static_cast<size_t>(count), std::vector<uint8_t>(static_cast<size_t>(17 * n * n)));
  for (auto& row : rows)
    for (uint8_t& v : row) v = static_cast<uint8_t>(rng.uniformInt(2));
  return rows;
}

std::vector<NNOutput> evaluateRows(TorchEvaluator& ev, const std::vector<std::vector<uint8_t>>& rows, size_t first, size_t count) {
  std::vector<NNInput> in;
  for (size_t i = first; i < first + count; ++i) in.push_back(NNInput{rows[i].data()});
  std::vector<NNOutput> out;
  ev.evaluate(in, out);
  return out;
}

double maxDiff(const std::vector<NNOutput>& a, const std::vector<NNOutput>& b) {
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].policy.size() == b[i].policy.size());
    for (size_t k = 0; k < a[i].policy.size(); ++k)
      worst = std::max(worst, static_cast<double>(std::fabs(a[i].policy[k] - b[i].policy[k])));
    worst = std::max(worst, static_cast<double>(std::fabs(a[i].value - b[i].value)));
  }
  return worst;
}

bool identical(const std::vector<NNOutput>& a, const std::vector<NNOutput>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].policy != b[i].policy || a[i].value != b[i].value) return false;
  return true;
}

TorchEvaluator::Options cudaOptions(bool half, bool channelsLast, bool graphs) {
  TorchEvaluator::Options o;
  o.device = "cuda";
  o.fp16 = half;
  o.channelsLast = channelsLast;
  o.cudaGraphs = graphs;
  return o;
}

}  // namespace

TEST_CASE("bucketFor rounds a batch up to its bucket") {
  CHECK(TorchEvaluator::bucketFor(1) == 1);
  CHECK(TorchEvaluator::bucketFor(3) == 4);
  CHECK(TorchEvaluator::bucketFor(16) == 16);
  CHECK(TorchEvaluator::bucketFor(33) == 48);
  CHECK(TorchEvaluator::bucketFor(100) == 128);
  CHECK(TorchEvaluator::bucketFor(1024) == 1024);
  CHECK(TorchEvaluator::bucketFor(1025) == 1536);
  CHECK(TorchEvaluator::bucketFor(4096) == 4096);
  CHECK(TorchEvaluator::bucketFor(4097) == 0);  // above the largest bucket: the plain path
  int previous = 0;
  for (int b = 1; b <= 4096; ++b) {
    const int k = TorchEvaluator::bucketFor(b);
    CHECK(k >= b);
    CHECK(k >= previous);
    CHECK(k < 2 * b);                        // never more than doubled
    if (b > 32) CHECK(k * 2 <= b * 3 + 2);  // from 32 on: padding wastes at most about a third of a forward
    previous = k;
  }
}

TEST_CASE("CUDA fast path: every switch combination passes the reference parity") {
  if (!TorchEvaluator::cudaAvailable()) {
    MESSAGE("CUDA not available: skipped");
    return;
  }
  Reference ref = loadReference(kFixture);
  REQUIRE(ref.inputs.size() >= 5);  // the batch-of-1 loop must get past the warm-up calls
  for (bool half : {false, true}) {
    for (bool channelsLast : {false, true}) {
      for (bool graphs : {false, true}) {
        CAPTURE(half);
        CAPTURE(channelsLast);
        CAPTURE(graphs);
        TorchEvaluator ev(kFixture, cudaOptions(half, channelsLast, graphs));
        CHECK(ev.isHalf() == half);
        CHECK(ev.channelsLast() == (channelsLast && half));  // fp16 only: NHWC fp32 kernels are not IEEE fp32
        CHECK(ev.cudaGraphs() == (graphs && TorchEvaluator::cudaGraphsCompiled()));
        checkAgainstReference(ev, ref, half);
        if (graphs && TorchEvaluator::cudaGraphsCompiled()) {
          CHECK(ev.cudaGraphs());            // no capture failed
          CHECK(ev.graphsCaptured() >= 1);   // bucket 1, from the batch-of-1 loop
          CHECK(ev.graphReplays() >= 1);
        } else {
          CHECK(ev.graphsCaptured() == 0);
          CHECK(ev.graphReplays() == 0);
        }
      }
    }
  }
}

TEST_CASE("CUDA fast path agrees with the plain path on and between buckets, warm-up and replay") {
  if (!TorchEvaluator::cudaAvailable() || !TorchEvaluator::cudaGraphsCompiled()) {
    MESSAGE("CUDA graphs not available: skipped");
    return;
  }
  const auto rows = randomPlanes(100, 5, 2024);
  for (bool half : {false, true}) {
    CAPTURE(half);
    const double tol = half ? 2e-3 : 1e-4;
    TorchEvaluator plain(kFixture, cudaOptions(half, false, false));
    TorchEvaluator fast(kFixture, cudaOptions(half, true, true));
    int captured = 0;
    for (size_t batch : {size_t{1}, size_t{5}, size_t{16}, size_t{33}, size_t{100}}) {
      CAPTURE(batch);
      const std::vector<NNOutput> expected = evaluateRows(plain, rows, 0, batch);
      const uint64_t replaysBefore = fast.graphReplays();
      std::vector<NNOutput> firstReplay;
      for (int call = 0; call < 6; ++call) {  // 3 warm-up forwards, then replays
        CAPTURE(call);
        const std::vector<NNOutput> got = evaluateRows(fast, rows, 0, batch);
        CHECK(maxDiff(got, expected) < tol);
        if (call == 3) firstReplay = got;
        if (call > 3) CHECK(identical(got, firstReplay));  // a replay is deterministic
      }
      CHECK(fast.graphReplays() == replaysBefore + 3);
      CHECK(fast.graphsCaptured() == ++captured);
      CHECK(fast.cudaGraphs());
    }
  }
}

TEST_CASE("CUDA fast path: static buffers do not leak between calls, buckets or evaluators") {
  if (!TorchEvaluator::cudaAvailable() || !TorchEvaluator::cudaGraphsCompiled()) {
    MESSAGE("CUDA graphs not available: skipped");
    return;
  }
  const auto rows = randomPlanes(120, 5, 77);
  for (bool half : {false, true}) {
    CAPTURE(half);
    const double tol = half ? 2e-3 : 1e-4;
    TorchEvaluator fast(kFixture, cudaOptions(half, true, true));
    // A (13 rows) and B (9 rows) share bucket 16; B's call leaves A's rows 9..12 stale in
    // the host buffer and the static input.
    for (int i = 0; i < 3; ++i) evaluateRows(fast, rows, 0, 13);
    REQUIRE(fast.graphsCaptured() == 1);
    const std::vector<NNOutput> a1 = evaluateRows(fast, rows, 0, 13);
    const std::vector<NNOutput> b = evaluateRows(fast, rows, 40, 9);
    const std::vector<NNOutput> a2 = evaluateRows(fast, rows, 0, 13);
    CHECK(fast.graphReplays() == 3);
    CHECK(identical(a1, a2));
    CHECK_FALSE(identical(a1, b));
    {
      TorchEvaluator fresh(kFixture, cudaOptions(half, false, false));
      CHECK(maxDiff(b, evaluateRows(fresh, rows, 40, 9)) < tol);
      CHECK(maxDiff(a1, evaluateRows(fresh, rows, 0, 13)) < tol);
    }
    // Buckets interleaved in one evaluator, and a second evaluator in the same process
    // (the match driver's situation): every call agrees with the plain path.
    TorchEvaluator plain(kFixture, cudaOptions(half, false, false));
    TorchEvaluator other(kFixture, cudaOptions(!half, true, true));
    TorchEvaluator otherPlain(kFixture, cudaOptions(!half, false, false));
    const double otherTol = !half ? 2e-3 : 1e-4;
    const size_t sizes[] = {3, 40, 3, 70, 40, 1, 70, 3, 40, 70, 1, 1, 3, 40, 70, 1};
    size_t first = 0;
    for (size_t batch : sizes) {
      CAPTURE(batch);
      CAPTURE(first);
      CHECK(maxDiff(evaluateRows(fast, rows, first, batch), evaluateRows(plain, rows, first, batch)) < tol);
      CHECK(maxDiff(evaluateRows(other, rows, first, batch), evaluateRows(otherPlain, rows, first, batch)) < otherTol);
      first = (first + 7) % 50;
    }
    CHECK(fast.cudaGraphs());
    CHECK(other.cudaGraphs());
    CHECK(fast.graphsCaptured() == 5);   // 16 from above, then 4, 48, 96, 1
    CHECK(other.graphsCaptured() == 4);  // 4, 48, 96, 1
  }
}

// Optional: the fast path against the plain path on a real model version (e.g. a 9x9
// 6x64 network of a run), on positions from random games. Set MANGO_TEST_MODEL_DIR to a
// model directory. The plain path's own sensitivity to the batch shape (the same rows
// evaluated one by one) is measured alongside: it is the scale the fast path's
// differences have to be read against.
#include "core/features.h"
#include "core/history.h"

namespace {

std::vector<std::vector<uint8_t>> gamePositions(int count, int n, float komi, uint64_t seed) {
  Rng rng(seed);
  std::vector<std::vector<uint8_t>> rows;
  std::vector<Move> legal;
  while (static_cast<int>(rows.size()) < count) {
    Board board(n, komi);
    GameHistory hist;
    hist.reset(board.hash());
    while (!board.gameOver() && static_cast<int>(rows.size()) < count) {
      if (rng.uniformInt(3) == 0) {  // every third position on average
        rows.emplace_back(static_cast<size_t>(featureSize(n)));
        encodeFeatures(board, rows.back().data());
      }
      board.legalMoves(HashHistory(hist), legal);  // pass is last
      const Move m = legal.size() == 1 ? kPass : legal[rng.uniformInt(static_cast<uint32_t>(legal.size() - 1))];
      board.play(m);
      hist.push(m, board.hash());
    }
  }
  return rows;
}

struct Diff {
  double policy = 0.0, value = 0.0;
};

Diff diffOf(const std::vector<NNOutput>& a, const std::vector<NNOutput>& b) {
  Diff d;
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t k = 0; k < a[i].policy.size(); ++k)
      d.policy = std::max(d.policy, static_cast<double>(std::fabs(a[i].policy[k] - b[i].policy[k])));
    d.value = std::max(d.value, static_cast<double>(std::fabs(a[i].value - b[i].value)));
  }
  return d;
}

}  // namespace

TEST_CASE("CUDA fast path agrees with the plain path on the model in MANGO_TEST_MODEL_DIR") {
  const char* dir = std::getenv("MANGO_TEST_MODEL_DIR");
  if (dir == nullptr || !TorchEvaluator::cudaAvailable() || !TorchEvaluator::cudaGraphsCompiled()) {
    MESSAGE("MANGO_TEST_MODEL_DIR not set (or no CUDA graphs): skipped");
    return;
  }
  const ModelMeta meta = ModelMeta::load(dir);
  const auto rows = gamePositions(1024, meta.boardSize, meta.komi, 31337);
  const size_t batches[] = {1, 7, 128, 200, 777, 1024};
  for (bool half : {false, true}) {
    CAPTURE(half);
    TorchEvaluator plain(dir, cudaOptions(half, false, false));
    // The plain path against itself: a batch of 200 and the same rows one by one.
    Diff shape;
    {
      const std::vector<NNOutput> together = evaluateRows(plain, rows, 0, 200);
      for (size_t i = 0; i < 200; ++i) {
        const Diff d = diffOf(evaluateRows(plain, rows, i, 1), {together[i]});
        shape.policy = std::max(shape.policy, d.policy);
        shape.value = std::max(shape.value, d.value);
      }
    }
    MESSAGE("model " << meta.modelId << (half ? " fp16" : " fp32") << ": plain path, batch 200 vs one by one: policy "
                     << shape.policy << " value " << shape.value);
    if (!half) {
      // Diagnostic: the plain path with TF32 allowed, against IEEE fp32.
      const std::vector<NNOutput> expected = evaluateRows(plain, rows, 0, 200);
      TorchEvaluator::Options o = cudaOptions(false, false, false);
      o.allowTf32 = true;
      Diff tf32;
      {
        TorchEvaluator withTf32(dir, o);
        tf32 = diffOf(evaluateRows(withTf32, rows, 0, 200), expected);
      }
      TorchEvaluator restore(dir, cudaOptions(false, false, false));  // the flag is process-wide
      MESSAGE("  plain path with TF32 allowed vs IEEE fp32: policy " << tf32.policy << " value " << tf32.value);
    }
    struct Variant {
      const char* name;
      bool channelsLast, graphs;
    };
    for (const Variant v : {Variant{"channels_last", true, false}, Variant{"graphs", false, true}, Variant{"both", true, true}}) {
      CAPTURE(v.name);
      TorchEvaluator fast(dir, cudaOptions(half, v.channelsLast, v.graphs));
      Diff worst;
      for (size_t batch : batches) {
        CAPTURE(batch);
        const std::vector<NNOutput> expected = evaluateRows(plain, rows, 0, batch);
        for (int call = 0; call < 5; ++call) {
          const Diff d = diffOf(evaluateRows(fast, rows, 0, batch), expected);
          worst.policy = std::max(worst.policy, d.policy);
          worst.value = std::max(worst.value, d.value);
        }
      }
      MESSAGE("  " << std::string(v.name) << " vs plain: policy " << worst.policy << " value " << worst.value);
      // fp32: IEEE fp32 on every path (channels-last is not applied). fp16: the fast
      // path may differ from the plain path by what a change of batch shape already
      // costs the plain path (twice that, at most), not by more.
      const double boundPolicy = half ? std::max(2e-3, 2.0 * shape.policy) : 1e-4;
      const double boundValue = half ? std::max(2e-3, 2.0 * shape.value) : 1e-4;
      CHECK(worst.policy < boundPolicy);
      CHECK(worst.value < boundValue);
      if (v.graphs) {
        CHECK(fast.cudaGraphs());
        CHECK(fast.graphsCaptured() == 5);  // buckets 1, 8, 128, 256, 1024 (777 pads to 1024)
      }
    }
  }
}
