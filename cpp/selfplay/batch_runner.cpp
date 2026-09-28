#include "selfplay/batch_runner.h"

#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "selfplay/eval_queue.h"

namespace mango {

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

}  // namespace

int DriverProfile::histogramBin(size_t batch) {
  int bin = 0;
  while (bin + 1 < kHistogramBins && batch >= (size_t{2} << bin)) ++bin;
  return bin;
}

void DriverProfile::recordBatch(size_t size, double atSeconds, int activeGames) {
  const int bin = histogramBin(size);
  histBatches[static_cast<size_t>(bin)] += 1;
  histEvaluations[static_cast<size_t>(bin)] += size;
  const size_t t = atSeconds > 0.0 ? static_cast<size_t>(atSeconds / kTimelineBinSeconds) : 0;
  if (timeline.size() <= t) timeline.resize(t + 1);
  timeline[t].batches += 1;
  timeline[t].evaluations += size;
  timeline[t].activeGamesSum += static_cast<uint64_t>(activeGames < 0 ? 0 : activeGames);
}

BatchedSelfplay::BatchedSelfplay(NNEvaluator& ev, int gamesInFlight, std::string modelId, int threads, int maxBatch)
    : ev_(ev),
      G_(gamesInFlight < 1 ? 1 : gamesInFlight),
      modelId_(std::move(modelId)),
      threads_(threads < 1 ? 1 : threads),
      maxBatch_(maxBatch) {
  if (threads_ > G_) threads_ = G_;  // a thread without a slot would have nothing to do
}

bool BatchedSelfplay::startGame(Slot& s, int& nextGameIndex, int totalGames, const OptionsFn& makeOptions,
                                BatchStats& stats, const DoneFn& onGameDone) {
  while (nextGameIndex < totalGames) {
    s.gameIndex = nextGameIndex++;
    s.game = std::make_unique<SelfplayGame>(makeOptions(s.gameIndex), modelId_);
    s.evaluations = 0;
    s.pending = false;
    if (s.game->finished()) {
      // Nothing to play (move cap 0): report the empty game like the sequential runner does.
      stats.games += 1;
      onGameDone(s.gameIndex, s.game->takeResult(0));
      continue;
    }
    s.game->tree().prepareRoot();  // the first move: nothing to do for a fresh root, kept symmetric
    return true;
  }
  s.game.reset();
  s.gameIndex = -1;
  return false;
}

bool BatchedSelfplay::collectForSlot(Slot& s, BatchStats& stats, const DoneFn& onGameDone) {
  // Mirrors runSequential: [prepareRoot] -> root expansion (not a simulation) ->
  // simulations until the budget is exhausted -> finishMove -> next move.
  for (;;) {
    SearchTree& tree = s.game->tree();
    if (!tree.rootExpanded()) {
      const CollectResult r = tree.collectLeaf(s.leaf);
      if (r == CollectResult::Pending) return true;
      if (r == CollectResult::Collision) throw std::logic_error("collision with K = 1");
      // Completed: the root itself was terminal (cannot happen: finishMove ends the game first).
      throw std::logic_error("terminal root reached in collect");
    }
    if (tree.budgetExhausted()) {
      const bool over = s.game->finishMove();
      if (over) {
        stats.games += 1;
        stats.positions += static_cast<uint64_t>(s.game->board().moveCount());
        onGameDone(s.gameIndex, s.game->takeResult(s.evaluations));
        return false;  // the caller starts a new game in this slot
      }
      tree.prepareRoot();  // reused root: noise, exactly as runSequential
      continue;
    }
    const CollectResult r = tree.collectLeaf(s.leaf);
    if (r == CollectResult::Pending) return true;
    if (r == CollectResult::Collision) throw std::logic_error("collision with K = 1");
    // Completed (terminal backup): loop for the next simulation.
  }
}

BatchStats BatchedSelfplay::run(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone) {
  if (threads_ == 1) return runSingleThread(totalGames, makeOptions, onGameDone);
  return runThreaded(totalGames, makeOptions, onGameDone);
}

// The first-version driver (M3b): one thread, one synchronous evaluator call per step.
BatchStats BatchedSelfplay::runSingleThread(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  BatchStats stats;
  DriverProfile& pf = stats.profile;
  pf.searchThreads = 1;
  double callbackSeconds = 0.0;
  const DoneFn timedDone = [&](int gameIndex, SelfplayGameResult&& r) {
    const auto c0 = clock::now();
    onGameDone(gameIndex, std::move(r));
    callbackSeconds += since(c0);
  };
  std::vector<Slot> slots(static_cast<size_t>(G_));
  int nextGameIndex = 0;
  int active = 0;
  auto start = [&](Slot& s) {
    const auto c0 = clock::now();
    const bool started = startGame(s, nextGameIndex, totalGames, makeOptions, stats, onGameDone);
    callbackSeconds += since(c0);
    return started;
  };
  for (Slot& s : slots)
    if (start(s)) ++active;

  std::vector<NNInput> in;
  std::vector<NNOutput> out;
  std::vector<Slot*> batch;
  while (active > 0) {
    batch.clear();
    const auto r0 = clock::now();
    const double callbacksBefore = callbackSeconds;
    for (Slot& s : slots) {
      if (!s.game) continue;
      // A slot keeps collecting until it has a leaf to evaluate; finished games are
      // replaced immediately so the batch stays full.
      while (s.game) {
        if (collectForSlot(s, stats, timedDone)) {
          s.pending = true;
          batch.push_back(&s);
          break;
        }
        --active;
        if (start(s)) ++active;
      }
    }
    pf.searchCollectSeconds += since(r0) - (callbackSeconds - callbacksBefore);
    if (batch.empty()) continue;
    in.clear();
    for (Slot* s : batch) in.push_back(s->leaf.input());
    const auto e0 = clock::now();
    try {
      ev_.evaluate(in, out);
    } catch (const std::exception&) {
      // Failure rule (DESIGN 5.4.5): retry the step once with the SAME requests (the
      // pending nodes, planes and symmetries are kept, so no RNG state is consumed and
      // the games stay reproducible from their seeds); on a second failure every
      // pending node reverts to Unexpanded, reservations are removed, completed
      // terminal simulations are kept, and the exception propagates.
      stats.retries += 1;
      try {
        ev_.evaluate(in, out);
      } catch (const std::exception&) {
        for (Slot* s : batch) {
          s->game->tree().abort(s->leaf);
          s->pending = false;
        }
        throw;
      }
    }
    stats.evalSeconds += std::chrono::duration<double>(clock::now() - e0).count();
    stats.batches += 1;
    stats.evaluations += static_cast<uint64_t>(batch.size());
    pf.recordBatch(batch.size(), since(t0), active);
    const auto m0 = clock::now();
    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->game->tree().commit(batch[i]->leaf, out[i]);
      batch[i]->evaluations += 1;
      batch[i]->pending = false;
    }
    pf.searchCommitSeconds += since(m0);
  }
  pf.searchCallbackSeconds = callbackSeconds;
  pf.rounds = stats.batches;
  pf.roundsInBatches = stats.batches;
  stats.seconds = std::chrono::duration<double>(clock::now() - t0).count();
  return stats;
}

// M4a driver (DESIGN 5.5.1): T search threads over the G slots, one evaluation thread.
// Per game nothing changes (collectForSlot / startGame are the first-version ones), so
// every game still equals its standalone sequential run under a deterministic evaluator.
BatchStats BatchedSelfplay::runThreaded(int totalGames, const OptionsFn& makeOptions, const DoneFn& onGameDone) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const int T = threads_;
  std::vector<Slot> slots(static_cast<size_t>(G_));
  EvalQueue queue(T, maxBatch_ <= 0 ? G_ : maxBatch_);

  // The callbacks and the game counter are shared: one mutex serialises them (the chunk
  // writer behind onGameDone appends in completion order).
  std::mutex callbackMutex;
  int nextGameIndex = 0;
  const DoneFn lockedDone = [&](int gameIndex, SelfplayGameResult&& r) {
    std::lock_guard<std::mutex> lk(callbackMutex);
    onGameDone(gameIndex, std::move(r));
  };
  auto startLocked = [&](Slot& s, BatchStats& st) {
    std::lock_guard<std::mutex> lk(callbackMutex);  // startGame may call onGameDone itself
    return startGame(s, nextGameIndex, totalGames, makeOptions, st, onGameDone);
  };

  std::mutex errorMutex;
  std::exception_ptr error;
  auto fail = [&](std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> lk(errorMutex);
      if (!error) error = std::move(e);
    }
    queue.abort();
  };

  std::atomic<int> activeGames{0};  // games in flight over all threads (profile only)
  std::vector<BatchStats> threadStats(static_cast<size_t>(T));
  // Result buffers outlive the search threads: after an abort the evaluation thread may
  // still write the results of a forward it had started into a finished thread's buffer.
  std::vector<std::vector<NNOutput>> resultBuffers(static_cast<size_t>(T));
  auto searchThread = [&](int t) {
    BatchStats& st = threadStats[static_cast<size_t>(t)];
    DriverProfile& pf = st.profile;
    double callbackSeconds = 0.0;
    const DoneFn timedDone = [&](int gameIndex, SelfplayGameResult&& r) {
      const auto c0 = clock::now();
      lockedDone(gameIndex, std::move(r));
      callbackSeconds += since(c0);
    };
    auto start = [&](Slot& s) {
      const auto c0 = clock::now();
      const bool started = startLocked(s, st);
      callbackSeconds += since(c0);
      if (started) activeGames.fetch_add(1, std::memory_order_relaxed);
      return started;
    };
    std::vector<Slot*> mine;
    for (int i = t; i < G_; i += T) mine.push_back(&slots[static_cast<size_t>(i)]);
    std::vector<NNOutput>& outs = resultBuffers[static_cast<size_t>(t)];
    outs.resize(mine.size());
    std::vector<EvalRequest> round;
    std::vector<std::pair<Slot*, NNOutput*>> pending;  // this round's slots and their results
    auto abandon = [&] {
      for (auto& [s, o] : pending) {
        s->game->tree().abort(s->leaf);
        s->pending = false;
      }
      pending.clear();
    };
    try {
      int active = 0;
      for (Slot* s : mine)
        if (start(*s)) ++active;
      while (active > 0 && !queue.aborted()) {
        round.clear();
        pending.clear();
        const auto r0 = clock::now();
        const double callbacksBefore = callbackSeconds;
        for (size_t k = 0; k < mine.size(); ++k) {
          Slot& s = *mine[k];
          while (s.game) {
            if (collectForSlot(s, st, timedDone)) {
              s.pending = true;
              round.push_back(EvalRequest{s.leaf.planes.data(), &outs[k], t, 0});
              pending.emplace_back(&s, &outs[k]);
              break;
            }
            --active;
            activeGames.fetch_sub(1, std::memory_order_relaxed);
            if (start(s)) ++active;
          }
        }
        pf.searchCollectSeconds += since(r0) - (callbackSeconds - callbacksBefore);
        if (round.empty()) continue;
        pf.rounds += 1;
        const auto w0 = clock::now();
        const bool delivered = queue.submitRound(t, round);
        pf.searchWaitSeconds += since(w0);
        if (!delivered) {
          abandon();
          pf.searchCallbackSeconds = callbackSeconds;
          return;
        }
        const auto m0 = clock::now();
        for (auto& [s, o] : pending) {
          s->game->tree().commit(s->leaf, *o);
          s->evaluations += 1;
          s->pending = false;
        }
        pf.searchCommitSeconds += since(m0);
        pending.clear();
      }
    } catch (...) {
      fail(std::current_exception());
      abandon();
    }
    pf.searchCallbackSeconds = callbackSeconds;
  };

  EvalThreadStats evalStats;
  DriverProfile evalProfile;  // written by the evaluation thread only
  auto evaluationThread = [&] {
    std::vector<EvalRequest> batch;
    std::vector<NNInput> in;
    std::vector<NNOutput> out;
    try {
      auto w0 = clock::now();
      while (queue.takeBatch(batch)) {
        const auto h0 = clock::now();
        evalProfile.evalWaitSeconds += std::chrono::duration<double>(h0 - w0).count();
        const double evalBefore = evalStats.evalSeconds;
        in.clear();
        for (const EvalRequest& r : batch) in.push_back(NNInput{r.planes});
        evaluateWithRetry(ev_, in, out, evalStats);
        for (size_t i = 0; i < batch.size(); ++i) *batch[i].out = std::move(out[i]);
        evalStats.batches += 1;
        evalStats.evaluations += static_cast<uint64_t>(batch.size());
        uint64_t rounds = 0;  // a thread has one round in the queue at a time, and rounds are contiguous
        for (size_t i = 0; i < batch.size(); ++i) rounds += (i == 0 || batch[i].thread != batch[i - 1].thread) ? 1 : 0;
        evalProfile.roundsInBatches += rounds;
        evalProfile.recordBatch(batch.size(), since(t0), activeGames.load(std::memory_order_relaxed));
        queue.deliver(batch);
        w0 = clock::now();
        evalProfile.evalHandbackSeconds += std::chrono::duration<double>(w0 - h0).count() - (evalStats.evalSeconds - evalBefore);
      }
      evalProfile.evalWaitSeconds += since(w0);
    } catch (...) {
      // Second failure of a forward (DESIGN 5.5.1): every search thread wakes, aborts its
      // pending leaves and exits; nothing of the affected batch is published.
      fail(std::current_exception());
    }
  };

  std::thread evaluator(evaluationThread);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) workers.emplace_back(searchThread, t);
  for (std::thread& w : workers) w.join();
  queue.stop();
  evaluator.join();
  if (error) std::rethrow_exception(error);

  BatchStats stats;
  stats.profile = std::move(evalProfile);
  stats.profile.searchThreads = T;
  for (const BatchStats& st : threadStats) {
    stats.games += st.games;
    stats.positions += st.positions;
    stats.profile.searchCollectSeconds += st.profile.searchCollectSeconds;
    stats.profile.searchCommitSeconds += st.profile.searchCommitSeconds;
    stats.profile.searchWaitSeconds += st.profile.searchWaitSeconds;
    stats.profile.searchCallbackSeconds += st.profile.searchCallbackSeconds;
    stats.profile.rounds += st.profile.rounds;
  }
  stats.evaluations = evalStats.evaluations;
  stats.batches = evalStats.batches;
  stats.retries = evalStats.retries;
  stats.evalSeconds = evalStats.evalSeconds;
  stats.seconds = std::chrono::duration<double>(clock::now() - t0).count();
  return stats;
}

}  // namespace mango
