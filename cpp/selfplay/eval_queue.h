// Request queue between T search threads and one evaluation thread (docs/DESIGN.md
// section 5.5.1, M4a). A search thread submits all its pending leaves as one round and
// blocks until every request of the round has its result; the evaluation thread takes
// up to `maxBatch` requests from the head in FIFO order (a round may be split over
// several forwards), runs one forward and delivers the results. A second consecutive
// evaluator failure, or an error in any thread, aborts the run: every waiting thread
// wakes, sees `aborted()` and abandons its pending leaves.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include "nn/evaluator.h"

namespace mango {

struct EvalRequest {
  const uint8_t* planes = nullptr;  // encoded features, owned by the submitting thread's PendingLeaf
  NNOutput* out = nullptr;          // where the result goes, owned by the submitting thread
  int thread = 0;                   // the round's owner
  int side = 0;                     // match driver: 0 = player A, 1 = player B (self-play: always 0)
};

// Counters of the evaluation thread (single writer; read after join).
struct EvalThreadStats {
  uint64_t batches = 0;      // forwards
  uint64_t evaluations = 0;  // requests evaluated
  uint64_t retries = 0;      // forwards repeated after a failure
  double evalSeconds = 0.0;  // inside evaluate(), retries included
};

class EvalQueue {
 public:
  // `maxBatch` <= 0 means no cap (every queued request fits one forward).
  EvalQueue(int threads, int maxBatch);

  // Search thread: enqueues the round and blocks until each of its requests has its
  // result. Returns false when the run was aborted meanwhile (the results are not to be
  // used; the caller aborts its pending leaves). An empty round returns at once.
  bool submitRound(int thread, const std::vector<EvalRequest>& round);

  // Evaluation thread: blocks until requests are queued, then moves up to `maxBatch` of
  // them (FIFO) into `batch`. Returns false once the queue is stopped and empty, or
  // aborted.
  bool takeBatch(std::vector<EvalRequest>& batch);

  // Evaluation thread: the results of `batch` are written; wakes every thread whose
  // round is now complete.
  void deliver(const std::vector<EvalRequest>& batch);

  // Any thread: abandons the run. Queued requests are dropped, every waiter wakes.
  void abort();
  // Driver: no further rounds will be submitted; takeBatch() returns false once empty.
  void stop();
  bool aborted() const;

 private:
  mutable std::mutex m_;
  std::condition_variable queued_;  // evaluation thread waits here
  std::condition_variable done_;    // search threads wait here
  std::deque<EvalRequest> q_;
  std::vector<int> outstanding_;    // per thread: requests of its round without a result yet
  int maxBatch_;
  bool abort_ = false;
  bool stop_ = false;
};

// One forward with the failure rule of DESIGN 5.4.5: a failed call is repeated once with
// the same requests (counted in `st.retries`); the second failure propagates. `out` is
// resized to in.size(). Timing is accumulated in `st.evalSeconds`.
void evaluateWithRetry(NNEvaluator& ev, const std::vector<NNInput>& in, std::vector<NNOutput>& out, EvalThreadStats& st);

}  // namespace mango
