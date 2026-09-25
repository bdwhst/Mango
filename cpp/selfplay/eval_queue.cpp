#include "selfplay/eval_queue.h"

#include <chrono>
#include <stdexcept>

namespace mango {

EvalQueue::EvalQueue(int threads, int maxBatch)
    : outstanding_(static_cast<size_t>(threads < 1 ? 1 : threads), 0), maxBatch_(maxBatch) {}

bool EvalQueue::submitRound(int thread, const std::vector<EvalRequest>& round) {
  std::unique_lock<std::mutex> lk(m_);
  if (abort_) return false;
  if (round.empty()) return true;
  if (outstanding_[static_cast<size_t>(thread)] != 0) throw std::logic_error("a thread submitted a round while one is outstanding");
  for (const EvalRequest& r : round) q_.push_back(r);
  outstanding_[static_cast<size_t>(thread)] = static_cast<int>(round.size());
  queued_.notify_one();
  done_.wait(lk, [&] { return outstanding_[static_cast<size_t>(thread)] == 0 || abort_; });
  return !abort_;
}

bool EvalQueue::takeBatch(std::vector<EvalRequest>& batch) {
  std::unique_lock<std::mutex> lk(m_);
  queued_.wait(lk, [&] { return !q_.empty() || stop_ || abort_; });
  batch.clear();
  if (abort_) {
    q_.clear();
    return false;
  }
  if (q_.empty()) return false;  // stopped and drained
  const size_t take = maxBatch_ <= 0 ? q_.size() : std::min(q_.size(), static_cast<size_t>(maxBatch_));
  for (size_t i = 0; i < take; ++i) {
    batch.push_back(q_.front());
    q_.pop_front();
  }
  return true;
}

void EvalQueue::deliver(const std::vector<EvalRequest>& batch) {
  std::lock_guard<std::mutex> lk(m_);
  for (const EvalRequest& r : batch) {
    int& left = outstanding_[static_cast<size_t>(r.thread)];
    if (left > 0) --left;
  }
  done_.notify_all();
}

void EvalQueue::abort() {
  std::lock_guard<std::mutex> lk(m_);
  abort_ = true;
  q_.clear();
  queued_.notify_all();
  done_.notify_all();
}

void EvalQueue::stop() {
  std::lock_guard<std::mutex> lk(m_);
  stop_ = true;
  queued_.notify_all();
}

bool EvalQueue::aborted() const {
  std::lock_guard<std::mutex> lk(m_);
  return abort_;
}

void evaluateWithRetry(NNEvaluator& ev, const std::vector<NNInput>& in, std::vector<NNOutput>& out, EvalThreadStats& st) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  try {
    ev.evaluate(in, out);
  } catch (const std::exception&) {
    // Failure rule (DESIGN 5.4.5): the same requests once more; the pending nodes,
    // planes and symmetries are kept, so no RNG state is consumed.
    st.retries += 1;
    try {
      ev.evaluate(in, out);
    } catch (const std::exception&) {
      st.evalSeconds += std::chrono::duration<double>(clock::now() - t0).count();
      throw;
    }
  }
  st.evalSeconds += std::chrono::duration<double>(clock::now() - t0).count();
  if (out.size() != in.size()) throw std::runtime_error("evaluator returned a wrong number of outputs");
}

}  // namespace mango
