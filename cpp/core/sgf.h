// Minimal SGF writer/reader: main line only, enough for self-play output, GTP
// debugging and tests.
#pragma once

#include <string>
#include <vector>

#include "core/types.h"

namespace mango {

struct SgfGame {
  int size = 9;
  float komi = 7.5f;
  std::vector<Move> moves;    // alternating, black first
  std::string result;         // e.g. "B+3.5", "W+R", "0"; empty if unknown
  std::string blackName;
  std::string whiteName;
  std::string comment;        // root GC property
};

std::string writeSgf(const SgfGame& g);

// Parses the main line of an SGF string. Returns false on malformed input.
bool parseSgf(const std::string& text, SgfGame* out);

// Result string helpers.
std::string resultString(float blackMinusWhiteScore, bool resigned, Color winnerIfResigned);

}  // namespace mango
