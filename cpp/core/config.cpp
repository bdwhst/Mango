#include "core/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/zobrist.h"
#include "nlohmann/json.hpp"

namespace mango {

using nlohmann::json;

namespace {

template <typename T>
void get(const json& j, const char* key, T& out) {
  if (j.contains(key)) out = j.at(key).get<T>();
}

}  // namespace

Config Config::fromJsonText(const std::string& text) {
  json j = json::parse(text);
  Config c;
  if (j.contains("board")) {
    const json& b = j["board"];
    get(b, "size", c.board.size);
    get(b, "komi", c.board.komi);
    get(b, "move_cap", c.board.moveCap);
  }
  if (j.contains("search")) {
    const json& s = j["search"];
    get(s, "simulations", c.search.simulations);
    get(s, "eval_simulations", c.search.evalSimulations);
    get(s, "c_puct", c.search.cPuct);
    get(s, "fpu", c.search.fpu);
    get(s, "dirichlet_alpha", c.search.dirichletAlpha);
    get(s, "dirichlet_epsilon", c.search.dirichletEpsilon);
    get(s, "temperature_moves", c.search.temperatureMoves);
    get(s, "search_symmetry", c.search.searchSymmetry);
    get(s, "tree_reuse", c.search.treeReuse);
    get(s, "budget_includes_inherited", c.search.budgetIncludesInherited);
    get(s, "resign_threshold", c.search.resignThreshold);
    get(s, "no_resign_fraction", c.search.noResignFraction);
    get(s, "nn_cache_size", c.search.nnCacheSize);
    get(s, "resolve_terminal_moves", c.search.resolveTerminalMoves);
  }
  if (j.contains("selfplay")) {
    const json& s = j["selfplay"];
    get(s, "games_per_iteration", c.selfplay.gamesPerIteration);
    get(s, "games_in_flight", c.selfplay.gamesInFlight);
    get(s, "leaves_per_game", c.selfplay.leavesPerGame);
    get(s, "chunk_games", c.selfplay.chunkGames);
    get(s, "save_sgf", c.selfplay.saveSgf);
  }
  if (j.contains("training")) {
    const json& t = j["training"];
    get(t, "res_blocks", c.training.resBlocks);
    get(t, "filters", c.training.filters);
    get(t, "batch_size", c.training.batchSize);
    get(t, "max_steps_per_iteration", c.training.maxStepsPerIteration);
    get(t, "samples_per_position", c.training.samplesPerPosition);
    get(t, "window_games", c.training.windowGames);
    get(t, "holdout_fraction", c.training.holdoutFraction);
    get(t, "l2", c.training.l2);
    get(t, "l2_all_params", c.training.l2AllParams);
    get(t, "store_pi", c.training.storePi);
    get(t, "lr0", c.training.lr0);
    get(t, "lr_step1", c.training.lrStep1);
    get(t, "lr_step2", c.training.lrStep2);
  }
  if (j.contains("eval")) {
    const json& e = j["eval"];
    get(e, "pairs", c.eval.pairs);
    get(e, "gate_threshold", c.eval.gateThreshold);
    get(e, "opening_moves", c.eval.openingMoves);
    get(e, "ladder_every", c.eval.ladderEvery);
    get(e, "gating", c.eval.gating);
  }
  if (c.board.size < 2 || c.board.size > 19) throw std::invalid_argument("board.size must be in [2, 19]");
  return c;
}

Config Config::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return fromJsonText(ss.str());
}

std::string Config::toJsonText() const {
  json j;
  j["board"] = {{"size", board.size}, {"komi", board.komi}, {"move_cap", board.moveCap}};
  j["search"] = {{"simulations", search.simulations},
                 {"eval_simulations", search.evalSimulations},
                 {"c_puct", search.cPuct},
                 {"fpu", search.fpu},
                 {"dirichlet_alpha", search.dirichletAlpha},
                 {"dirichlet_epsilon", search.dirichletEpsilon},
                 {"temperature_moves", search.temperatureMoves},
                 {"search_symmetry", search.searchSymmetry},
                 {"tree_reuse", search.treeReuse},
                 {"budget_includes_inherited", search.budgetIncludesInherited},
                 {"resign_threshold", search.resignThreshold},
                 {"no_resign_fraction", search.noResignFraction},
                 {"nn_cache_size", search.nnCacheSize},
                 {"resolve_terminal_moves", search.resolveTerminalMoves}};
  j["selfplay"] = {{"games_per_iteration", selfplay.gamesPerIteration},
                   {"games_in_flight", selfplay.gamesInFlight},
                   {"leaves_per_game", selfplay.leavesPerGame},
                   {"chunk_games", selfplay.chunkGames},
                   {"save_sgf", selfplay.saveSgf}};
  j["training"] = {{"res_blocks", training.resBlocks},
                   {"filters", training.filters},
                   {"batch_size", training.batchSize},
                   {"max_steps_per_iteration", training.maxStepsPerIteration},
                   {"samples_per_position", training.samplesPerPosition},
                   {"window_games", training.windowGames},
                   {"holdout_fraction", training.holdoutFraction},
                   {"l2", training.l2},
                   {"l2_all_params", training.l2AllParams},
                   {"store_pi", training.storePi},
                   {"lr0", training.lr0},
                   {"lr_step1", training.lrStep1},
                   {"lr_step2", training.lrStep2}};
  j["eval"] = {{"pairs", eval.pairs},
               {"gate_threshold", eval.gateThreshold},
               {"opening_moves", eval.openingMoves},
               {"ladder_every", eval.ladderEvery},
               {"gating", eval.gating}};
  return j.dump(2);
}

std::string Config::fingerprint() const {
  // Fingerprint of the sections that determine self-play data semantics. A cheap,
  // deterministic 64-bit hash of the canonical JSON is enough for consistency checks.
  json j;
  j["board"] = {{"size", board.size}, {"komi", board.komi}, {"move_cap", board.effectiveMoveCap()}};
  j["search"] = {{"simulations", search.simulations},
                 {"c_puct", search.cPuct},
                 {"fpu", search.fpu},
                 {"dirichlet_alpha", search.effectiveDirichletAlpha(board.size)},
                 {"dirichlet_epsilon", search.dirichletEpsilon},
                 {"temperature_moves", search.temperatureMoves},
                 {"search_symmetry", search.searchSymmetry},
                 {"tree_reuse", search.treeReuse},
                 {"budget_includes_inherited", search.budgetIncludesInherited},
                 {"resolve_terminal_moves", search.resolveTerminalMoves}};
  j["rules_id"] = 1;
  j["feature_schema"] = 1;
  const std::string canon = j.dump();
  uint64_t h = 0xCBF29CE484222325ull;
  for (unsigned char ch : canon) {
    h ^= ch;
    h *= 0x100000001B3ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

}  // namespace mango
