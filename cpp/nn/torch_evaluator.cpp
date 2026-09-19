#include "nn/torch_evaluator.h"

#include <cstring>
#include <stdexcept>

#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/script.h>

namespace mango {

struct TorchEvaluator::Impl {
  torch::jit::Module module;
  torch::Device device{torch::kCPU};
  torch::ScalarType dtype = torch::kFloat32;
  std::vector<torch::jit::IValue> inputs;
};

bool TorchEvaluator::cudaAvailable() { return torch::cuda::is_available(); }
bool TorchEvaluator::mpsAvailable() { return torch::mps::is_available(); }

TorchEvaluator::TorchEvaluator(const std::string& modelDir, const Options& options)
    : meta_(ModelMeta::load(modelDir)), impl_(std::make_unique<Impl>()) {
  std::string dev = options.device;
  if (dev == "auto") {
    if (cudaAvailable()) dev = "cuda";
    else if (mpsAvailable()) dev = "mps";
    else dev = "cpu";
  }
  if (dev == "cuda") {
    if (!cudaAvailable()) throw std::runtime_error("CUDA requested but not available");
    impl_->device = torch::Device(torch::kCUDA);
    // Process-wide flags (there is one ATen context); every evaluator in a process sets
    // them the same way, so the last constructed one wins harmlessly.
    at::globalContext().setAllowTF32CuDNN(options.allowTf32);
    at::globalContext().setAllowTF32CuBLAS(options.allowTf32);
  } else if (dev == "mps") {
    if (!mpsAvailable()) throw std::runtime_error("MPS requested but not available");
    impl_->device = torch::Device(torch::kMPS);
  } else if (dev == "cpu") {
    impl_->device = torch::Device(torch::kCPU);
  } else {
    throw std::runtime_error("unknown device: " + dev);
  }
  c10::InferenceMode guard;
  impl_->module = torch::jit::load(modelDir + "/model.pt", impl_->device);
  impl_->module.eval();
  if (dev == "cuda" && options.fp16) {
    impl_->module.to(torch::kHalf);
    impl_->dtype = torch::kHalf;
  }
}

TorchEvaluator::~TorchEvaluator() = default;

std::string TorchEvaluator::deviceName() const { return impl_->device.str(); }
bool TorchEvaluator::isHalf() const { return impl_->dtype == torch::kHalf; }

void TorchEvaluator::evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) {
  const int n = meta_.boardSize;
  const int nn = n * n;
  const int64_t batch = static_cast<int64_t>(in.size());
  out.resize(in.size());
  if (batch == 0) return;

  c10::InferenceMode guard;
  torch::Tensor host = torch::empty({batch, kNumPlanes, n, n}, torch::kUInt8);
  uint8_t* dst = host.data_ptr<uint8_t>();
  const size_t per = static_cast<size_t>(kNumPlanes) * nn;
  for (int64_t i = 0; i < batch; ++i) std::memcpy(dst + i * per, in[i].planes, per);

  torch::Tensor x = host.to(impl_->device, /*non_blocking=*/false).to(impl_->dtype);
  impl_->inputs.clear();
  impl_->inputs.emplace_back(x);
  auto result = impl_->module.forward(impl_->inputs).toTuple();
  torch::Tensor logits = result->elements()[0].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
  torch::Tensor values = result->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (logits.size(0) != batch || logits.size(1) != nn + 1 || values.numel() != batch)
    throw std::runtime_error("unexpected output shapes from model");

  // fp32 softmax on the CPU (DESIGN 5.3): even a half model cannot underflow priors to 0 here.
  const float* lp = logits.data_ptr<float>();
  const float* vp = values.data_ptr<float>();
  const int na = nn + 1;
  for (int64_t i = 0; i < batch; ++i) {
    const float* row = lp + i * na;
    float mx = row[0];
    for (int a = 1; a < na; ++a) mx = row[a] > mx ? row[a] : mx;
    out[i].policy.resize(na);
    double sum = 0.0;
    for (int a = 0; a < na; ++a) {
      double e = std::exp(static_cast<double>(row[a] - mx));
      out[i].policy[a] = static_cast<float>(e);
      sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int a = 0; a < na; ++a) out[i].policy[a] *= inv;
    out[i].value = vp[i];
  }
}

}  // namespace mango
