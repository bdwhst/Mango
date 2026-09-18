// LibTorch (TorchScript) implementation of NNEvaluator. The only file that knows
// about devices; CUDA / MPS / CPU are selected at runtime.
#pragma once

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
  };

  TorchEvaluator(const std::string& modelDir, const Options& options);
  ~TorchEvaluator() override;

  int boardSize() const override { return meta_.boardSize; }
  const std::string& modelId() const override { return meta_.modelId; }
  const ModelMeta& meta() const { return meta_; }
  std::string deviceName() const;
  bool isHalf() const;

  void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override;

  static bool cudaAvailable();
  static bool mpsAvailable();

 private:
  struct Impl;
  ModelMeta meta_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mango
