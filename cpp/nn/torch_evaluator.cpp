#include "nn/torch_evaluator.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

#ifdef MANGO_CUDA_GRAPHS
// CUDA_VERSION (from cuda.h) must be defined before CUDAGraph.h: the layout of
// at::cuda::CUDAGraph depends on it, and it has to match the library's.
#include <cuda.h>

#include <ATen/cuda/CUDAGraph.h>
#endif
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/script.h>

namespace mango {

namespace {

// Batch buckets of the CUDA-graph path (DESIGN 5.3): powers of two, and their 1.5x
// midpoints from 32 on, so padding wastes at most a third of a forward.
const int kBuckets[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};

// Process-wide: every evaluator holds it shared while it uses the GPU (construction,
// evaluate, destruction) and a graph capture holds it exclusively. PyTorch captures in
// the global mode, in which a CUDA call of any other thread during the capture
// invalidates it — a real race as soon as evaluators are used from two threads at once.
std::shared_mutex& cudaWorkMutex() {
  static std::shared_mutex m;
  return m;
}

}  // namespace

struct TorchEvaluator::Impl {
  torch::jit::Module module;
  torch::Device device{torch::kCPU};
  torch::ScalarType dtype = torch::kFloat32;
  std::vector<torch::jit::IValue> inputs;
  int boardSize = 0;
  bool channelsLast = false;
  bool graphs = false;
  int warmup = 3;
  uint64_t replays = 0;
  TorchEvaluator::Timings timings;

  struct Bucket {
    int size = 0;
    int plainCalls = 0;
    torch::Tensor host;       // uint8 [B,17,n,n] on the CPU; rows beyond the batch keep stale planes
    torch::Tensor staticIn;   // the graph's input (device, uint8)
    torch::Tensor staticOut;  // the graph's output (device, fp32 [B, n*n+2])
#ifdef MANGO_CUDA_GRAPHS
    std::unique_ptr<at::cuda::CUDAGraph> graph;
#endif
    bool captured = false;
  };
  std::map<int, Bucket> buckets;

  // uint8 planes on the device -> [B, n*n+2] fp32 on the device: the logits followed by
  // the value, so one copy brings both back.
  torch::Tensor forwardPacked(const torch::Tensor& planes) {
    const int64_t na = static_cast<int64_t>(boardSize) * boardSize + 1;
    torch::Tensor x = planes.to(dtype);
    if (channelsLast) x = x.contiguous(at::MemoryFormat::ChannelsLast);
    inputs.clear();
    inputs.emplace_back(x);
    auto result = module.forward(inputs).toTuple();
    torch::Tensor logits = result->elements()[0].toTensor();
    torch::Tensor values = result->elements()[1].toTensor();
    if (logits.dim() != 2 || logits.size(0) != planes.size(0) || logits.size(1) != na || values.numel() != planes.size(0))
      throw std::runtime_error("unexpected output shapes from model");
    return torch::cat({logits.to(torch::kFloat32), values.to(torch::kFloat32).reshape({-1, 1})}, 1);
  }

  Bucket& bucket(int size) {
    auto it = buckets.find(size);
    if (it != buckets.end()) return it->second;
    Bucket& b = buckets[size];
    b.size = size;
    b.host = torch::zeros({size, kNumPlanes, boardSize, boardSize}, torch::kUInt8);
    return b;
  }

  // Captures the bucket's graph. On any failure the graph path is switched off for this
  // evaluator (the plain path is always correct) and the reason is reported once.
  void capture(Bucket& b) {
#ifdef MANGO_CUDA_GRAPHS
    try {
      b.staticIn = b.host.to(device);
      torch::cuda::synchronize();
      auto g = std::make_unique<at::cuda::CUDAGraph>();
      {
        // A graph must be captured on a non-default stream; replaying on the default
        // stream afterwards is fine.
        c10::cuda::CUDAStreamGuard onCaptureStream(c10::cuda::getStreamFromPool());
        g->capture_begin();
        try {
          b.staticOut = forwardPacked(b.staticIn);
        } catch (...) {
          try {
            g->capture_end();
          } catch (...) {
          }
          throw;
        }
        g->capture_end();
      }
      torch::cuda::synchronize();
      b.graph = std::move(g);
      b.captured = true;
    } catch (const std::exception& e) {
      std::cerr << "mango: CUDA graph capture failed for batch " << b.size << " (" << e.what()
                << "); using the plain forward path\n";
      disableGraphs();
    }
#else
    (void)b;
    disableGraphs();
#endif
  }

  void disableGraphs() {
    graphs = false;
    buckets.clear();
  }
};

bool TorchEvaluator::cudaAvailable() { return torch::cuda::is_available(); }
bool TorchEvaluator::mpsAvailable() { return torch::mps::is_available(); }
bool TorchEvaluator::cudaGraphsCompiled() {
#ifdef MANGO_CUDA_GRAPHS
  return true;
#else
  return false;
#endif
}

int TorchEvaluator::bucketFor(int batch) {
  for (int b : kBuckets)
    if (b >= batch) return b;
  return 0;
}

TorchEvaluator::TorchEvaluator(const std::string& modelDir, const Options& options)
    : meta_(ModelMeta::load(modelDir)), impl_(std::make_unique<Impl>()) {
  std::shared_lock<std::shared_mutex> gpuWork(cudaWorkMutex());
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
  impl_->boardSize = meta_.boardSize;
  c10::InferenceMode guard;
  impl_->module = torch::jit::load(modelDir + "/model.pt", impl_->device);
  impl_->module.eval();
  if (dev == "cuda" && options.fp16) {
    impl_->module.to(torch::kHalf);
    impl_->dtype = torch::kHalf;
  }
  if (dev == "cuda") {
    if (options.channelsLast && impl_->dtype == torch::kHalf) {
      // Convolution weights in the channels-last memory format: cuDNN then runs its NHWC
      // kernels on channels-last activations without transposing around every call.
      // fp16 only: cuDNN's NHWC fp32 kernels compute at TF32-class precision even with
      // TF32 switched off (measured on a 9x9 6x64 model: 1.4e-3 against IEEE fp32, the
      // same as allowTf32), and "fp32" has to stay IEEE fp32 (DESIGN 5.3).
      for (const auto& p : impl_->module.parameters()) {
        if (p.dim() == 4) {
          torch::Tensor t = p;
          t.set_data(p.contiguous(at::MemoryFormat::ChannelsLast));
        }
      }
      impl_->channelsLast = true;
    }
    impl_->graphs = options.cudaGraphs && cudaGraphsCompiled();
    impl_->warmup = options.graphWarmup < 1 ? 1 : options.graphWarmup;
  }
}

TorchEvaluator::~TorchEvaluator() {
  std::shared_lock<std::shared_mutex> gpuWork(cudaWorkMutex());
  impl_.reset();
}

std::string TorchEvaluator::deviceName() const { return impl_->device.str(); }
bool TorchEvaluator::isHalf() const { return impl_->dtype == torch::kHalf; }
bool TorchEvaluator::channelsLast() const { return impl_->channelsLast; }
bool TorchEvaluator::cudaGraphs() const { return impl_->graphs; }
uint64_t TorchEvaluator::graphReplays() const { return impl_->replays; }
const TorchEvaluator::Timings& TorchEvaluator::timings() const { return impl_->timings; }
int TorchEvaluator::graphsCaptured() const {
  int n = 0;
  for (const auto& [size, b] : impl_->buckets) n += b.captured ? 1 : 0;
  return n;
}
std::string TorchEvaluator::description() const {
  std::string s = deviceName() + (isHalf() ? " fp16" : " fp32");
  if (impl_->channelsLast) s += " channels_last";
  if (impl_->graphs) s += " graphs";
  return s;
}

void TorchEvaluator::evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) {
  const int n = meta_.boardSize;
  const int nn = n * n;
  const int64_t batch = static_cast<int64_t>(in.size());
  out.resize(in.size());
  if (batch == 0) return;

  c10::InferenceMode guard;
  std::shared_lock<std::shared_mutex> gpuWork(cudaWorkMutex());
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  auto t1 = t0;  // end of packing
  Impl::Bucket* toCapture = nullptr;
  const size_t per = static_cast<size_t>(kNumPlanes) * nn;
  torch::Tensor packed;  // [>= batch, nn + 2] fp32 on the CPU: logits, then the value
  const int bucketSize = impl_->graphs ? bucketFor(static_cast<int>(batch)) : 0;
  if (bucketSize > 0) {
    Impl::Bucket& b = impl_->bucket(bucketSize);
    uint8_t* dst = b.host.data_ptr<uint8_t>();
    for (int64_t i = 0; i < batch; ++i) std::memcpy(dst + i * per, in[i].planes, per);
    t1 = clock::now();
    if (b.captured) {
#ifdef MANGO_CUDA_GRAPHS
      b.staticIn.copy_(b.host);
      b.graph->replay();
      packed = b.staticOut.to(torch::kCPU);
      impl_->replays += 1;
#endif
    } else {
      // The bucket's shape through the plain path until the JIT and cuDNN have settled.
      packed = impl_->forwardPacked(b.host.to(impl_->device)).to(torch::kCPU);
      if (++b.plainCalls >= impl_->warmup) toCapture = &b;  // captured at the end, exclusively
    }
  } else {
    torch::Tensor host = torch::empty({batch, kNumPlanes, n, n}, torch::kUInt8);
    uint8_t* dst = host.data_ptr<uint8_t>();
    for (int64_t i = 0; i < batch; ++i) std::memcpy(dst + i * per, in[i].planes, per);
    t1 = clock::now();
    packed = impl_->forwardPacked(host.to(impl_->device, /*non_blocking=*/false)).to(torch::kCPU);
  }
  packed = packed.contiguous();
  const auto t2 = clock::now();
  if (packed.size(0) < batch || packed.size(1) != nn + 2) throw std::runtime_error("unexpected output shapes from model");

  // fp32 softmax on the CPU (DESIGN 5.3): even a half model cannot underflow priors to 0 here.
  const float* pp = packed.data_ptr<float>();
  const int na = nn + 1;
  for (int64_t i = 0; i < batch; ++i) {
    const float* row = pp + i * (na + 1);
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
    out[i].value = row[na];
  }
  Timings& tm = impl_->timings;
  tm.calls += 1;
  tm.rows += static_cast<uint64_t>(batch);
  tm.paddedRows += static_cast<uint64_t>(bucketSize > 0 ? bucketSize : batch);
  tm.packSeconds += std::chrono::duration<double>(t1 - t0).count();
  tm.deviceSeconds += std::chrono::duration<double>(t2 - t1).count();
  tm.unpackSeconds += std::chrono::duration<double>(clock::now() - t2).count();
  if (toCapture != nullptr) {
    // A capture must not overlap any other CUDA work of the process (another evaluator's
    // forward on another thread invalidates it): wait until every evaluator is between
    // calls.
    gpuWork.unlock();
    std::unique_lock<std::shared_mutex> exclusive(cudaWorkMutex());
    impl_->capture(*toCapture);
  }
}

}  // namespace mango
