// Basic types and geometry shared by the whole engine.
#pragma once

#include <cstdint>

namespace mango {

constexpr int kMaxN = 19;                       // largest supported board
constexpr int kMinN = 2;
constexpr int kStride = kMaxN + 2;              // bordered row stride (21)
constexpr int kMaxCells = kStride * kStride;    // bordered cells (441)
constexpr int kMaxPoints = kMaxN * kMaxN;       // unbordered points (361)
constexpr int kNumPlanes = 17;                  // AGZ feature planes
constexpr int kHistoryLen = 8;                  // history steps per colour in the features
constexpr int kFeatureSchema = 1;               // features_v1
constexpr int kRulesId = 1;                     // TT scoring, no suicide, positional superko

enum class Color : uint8_t { Empty = 0, Black = 1, White = 2, Wall = 3 };

inline Color opposite(Color c) {
  if (c == Color::Black) return Color::White;
  if (c == Color::White) return Color::Black;
  return c;
}
inline int colorIndex(Color c) { return c == Color::Black ? 0 : 1; }  // snapshot / zobrist index
inline char colorChar(Color c) {
  switch (c) {
    case Color::Empty: return '.';
    case Color::Black: return 'X';
    case Color::White: return 'O';
    default: return '#';
  }
}

// A move is an unbordered point index in [0, n*n), or kPass.
using Move = int16_t;
constexpr Move kPass = -1;
constexpr Move kNoMove = -2;

// Geometry helpers. "loc" = bordered index, "point" = unbordered index r*n+c.
inline int locOf(int r, int c) { return (r + 1) * kStride + (c + 1); }
inline int pointOf(int r, int c, int n) { return r * n + c; }
inline int rowOfPoint(int p, int n) { return p / n; }
inline int colOfPoint(int p, int n) { return p % n; }
inline int locOfPoint(int p, int n) { return locOf(p / n, p % n); }
inline int pointOfLoc(int loc, int n) { return (loc / kStride - 1) * n + (loc % kStride - 1); }
inline int rowOfLoc(int loc) { return loc / kStride - 1; }
inline int colOfLoc(int loc) { return loc % kStride - 1; }

constexpr int kDirs[4] = {1, -1, kStride, -kStride};

}  // namespace mango
