#include "selfplay/batch_runner.h"

#include <stdexcept>

namespace mango {

BatchedSelfplay::BatchedSelfplay(NNEvaluator& ev, int gamesInFlight, std::string modelId)
    : ev_(ev), G_(gamesInFlight < 1 ? 1 : gamesInFlight), modelId_(std::move(modelId)) {}

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

}  // namespace mango
