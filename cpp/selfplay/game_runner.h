// One self-play game with the sequential search (docs/DESIGN.md sections 5.5, 5.4.6,
// 5.4.7): root noise, temperature moves, move cap, optional resignation, and the
// per-move data the chunk stores. Evaluator-agnostic so tests can use FakeEvaluator.
#pragma once

#include <cstdint>
#include <string>

#include "core/board.h"
#include "core/sgf.h"
#include "nn/evaluator.h"
#include "search/search_params.h"
#include "selfplay/chunk.h"

namespace mango {

struct SelfplayGameOptions {
  int boardSize = 9;
  float komi = 7.5f;
  int moveCap = -1;           // <0: 2*n*n
  SearchParams params;        // self-play params (noise on, temperature moves, resign threshold)
  uint64_t gameSeed = 0;      // drives noise, temperature sampling, symmetries
  bool noResignGame = false;  // play out even if resignation is enabled
  bool storeFinalOwnership = false;  // record_extras bit 0
  bool storeSearchKind = false;      // record_extras bit 1 (always 1 in this driver: full searches)
};

struct SelfplayGameResult {
  GameRecord record;
  SgfGame sgf;
  int evaluations = 0;  // positions sent to the network
};

// Plays a complete game from the empty board and returns the record. The record's
// gameIndex is left 0 (the ChunkWriter assigns it).
SelfplayGameResult playSelfplayGame(NNEvaluator& ev, const SelfplayGameOptions& opt, const std::string& modelId);

// Whether a game with this seed is a no-resign game under `fraction` (deterministic).
bool isNoResignGame(uint64_t gameSeed, float fraction);

}  // namespace mango
