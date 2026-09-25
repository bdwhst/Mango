#include "selfplay/batch_runner.h"

#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "selfplay/eval_queue.h"

namespace mango {

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
  std::vector<Slot> slots(static_cast<size_t>(G_));
  int nextGameIndex = 0;
  int active = 0;
  for (Slot& s : slots)
    if (startGame(s, nextGameIndex, totalGames, makeOptions, stats, onGameDone)) ++active;

  std::vector<NNInput> in;
  std::vector<NNOutput> out;
  std::vector<Slot*> batch;
  while (active > 0) {
    batch.clear();
    for (Slot& s : slots) {
      if (!s.game) continue;
      // A slot keeps collecting until it has a leaf to evaluate; finished games are
      // replaced immediately so the batch stays full.
      while (s.game) {
        if (collectForSlot(s, stats, onGameDone)) {
          s.pending = true;
          batch.push_back(&s);
          break;
        }
        --active;
        if (startGame(s, nextGameIndex, totalGames, makeOptions, stats, onGameDone)) ++active;
      }
    }
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
    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->game->tree().commit(batch[i]->leaf, out[i]);
      batch[i]->evaluations += 1;
      batch[i]->pending = false;
    }
  }
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

  std::vector<BatchStats> threadStats(static_cast<size_t>(T));
  auto searchThread = [&](int t) {
    BatchStats& st = threadStats[static_cast<size_t>(t)];
    std::vector<Slot*> mine;
    for (int i = t; i < G_; i += T) mine.push_back(&slots[static_cast<size_t>(i)]);
    std::vector<NNOutput> outs(mine.size());
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
        if (startLocked(*s, st)) ++active;
      while (active > 0 && !queue.aborted()) {
        round.clear();
        pending.clear();
        for (size_t k = 0; k < mine.size(); ++k) {
          Slot& s = *mine[k];
          while (s.game) {
            if (collectForSlot(s, st, lockedDone)) {
              s.pending = true;
              round.push_back(EvalRequest{s.leaf.planes.data(), &outs[k], t, 0});
              pending.emplace_back(&s, &outs[k]);
              break;
            }
            --active;
            if (startLocked(s, st)) ++active;
          }
        }
        if (round.empty()) continue;
        if (!queue.submitRound(t, round)) {
          abandon();
          return;
        }
        for (auto& [s, o] : pending) {
          s->game->tree().commit(s->leaf, *o);
          s->evaluations += 1;
          s->pending = false;
        }
        pending.clear();
      }
    } catch (...) {
      fail(std::current_exception());
      abandon();
    }
  };

  EvalThreadStats evalStats;
  auto evaluationThread = [&] {
    std::vector<EvalRequest> batch;
    std::vector<NNInput> in;
    std::vector<NNOutput> out;
    try {
      while (queue.takeBatch(batch)) {
        in.clear();
        for (const EvalRequest& r : batch) in.push_back(NNInput{r.planes});
        evaluateWithRetry(ev_, in, out, evalStats);
        for (size_t i = 0; i < batch.size(); ++i) *batch[i].out = std::move(out[i]);
        evalStats.batches += 1;
        evalStats.evaluations += static_cast<uint64_t>(batch.size());
        queue.deliver(batch);
      }
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
  for (const BatchStats& st : threadStats) {
    stats.games += st.games;
    stats.positions += st.positions;
  }
  stats.evaluations = evalStats.evaluations;
  stats.batches = evalStats.batches;
  stats.retries = evalStats.retries;
  stats.evalSeconds = evalStats.evalSeconds;
  stats.seconds = std::chrono::duration<double>(clock::now() - t0).count();
  return stats;
}

}  // namespace mango
