// LibTorch (TorchScript) implementation of NNEvaluator. The only file that knows
// about devices; CUDA / MPS / CPU are selected at runtime.
//
// On CUDA the evaluator has a fast path (docs/DESIGN.md section 5.3, M4a step 2):
// channels-last parameters and input (no NCHW<->NHWC transposes around the fp16
// convolutions) and one CUDA graph per batch bucket (one launch per forward instead of
// one per kernel). Both are switches; with both off the evaluator is the M2 one.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nn/evaluator.h"
#include "nn/model_meta.h"

namespace mango {

class TorchEvaluator : public NNEvaluator {
 public:
  struct Options {
    std::string device = "auto";  // auto | cuda | mps | cpu
    bool fp16 = true;             // only honoured on CUDA
    // CUDA fp32 only: let cuDNN/cuBLAS use TF32 tensor cores (10-bit mantissa) for fp32 convs
    // and matmuls. LibTorch enables TF32 for cuDNN by default; we turn it off so "fp32"
    // means IEEE fp32 and matches the PyTorch reference to 1e-4. Irrelevant under fp16.
    bool allowTf32 = false;
    // CUDA only (ignored elsewhere): the fast path of DESIGN 5.3. channelsLast takes
    // effect with fp16 only (the NHWC fp32 kernels are not IEEE fp32).
    bool channelsLast = true;
    bool cudaGraphs = true;
    int graphWarmup = 3;  // plain forwards of a bucket's shape before its graph is captured
  };

  TorchEvaluator(const std::string& modelDir, const Options& options);
  ~TorchEvaluator() override;

  int boardSize() const override { return meta_.boardSize; }
  const std::string& modelId() const override { return meta_.modelId; }
  const ModelMeta& meta() const { return meta_; }
  std::string deviceName() const;
  bool isHalf() const;
  // "cuda fp16 channels_last graphs" etc.: device, precision and the active switches.
  std::string description() const;

  void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override;

  // Fast-path state (false / 0 on MPS and CPU, or when switched off).
  bool channelsLast() const;
  bool cudaGraphs() const;       // still enabled (a failed capture disables it)
  int graphsCaptured() const;    // buckets with a captured graph
  uint64_t graphReplays() const; // forwards served by a replay
  // The bucket a batch of `batch` positions is padded to; 0 when it exceeds the largest
  // bucket (such batches use the plain path).
  static int bucketFor(int batch);

  static bool cudaAvailable();
  static bool mpsAvailable();
  // Whether this build can capture CUDA graphs at all (compiled with the CUDA headers).
  static bool cudaGraphsCompiled();

 private:
  struct Impl;
  ModelMeta meta_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mango
