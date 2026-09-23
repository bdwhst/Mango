// MGO2 game-chunk format (docs/DESIGN.md section 5.6): per-game records with packed
// snapshots, moves, root statistics and sparse visit counts. Every field is written
// and read one by one, little-endian, at the offsets given in the design; no struct
// is ever copied as a whole.
#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"

namespace mango {

constexpr int kChunkHeaderSize = 128;
constexpr uint8_t kChunkVersion = 2;

// record_extras bits (header offset 100).
constexpr uint8_t kExtrasFinalOwnership = 1;  // every game record ends with N^2 int8
constexpr uint8_t kExtrasSearchKind = 2;      // per-move u8 after the moves array
constexpr uint8_t kExtrasPrunedVisits = 4;    // stored visit counts are pruned: sum <= root_total

enum class Termination : uint8_t { TwoPasses = 0, Resign = 1, MoveCap = 2 };

struct ChunkHeader {
  int boardSize = 0;
  uint8_t planes = kNumPlanes;
  uint8_t featureSchema = kFeatureSchema;
  uint8_t rulesId = kRulesId;
  float komi = 7.5f;
  uint32_t numGames = 0;      // filled by the writer
  uint32_t numPositions = 0;  // sum of T, filled by the writer
  uint32_t simulationsPerMove = 0;
  std::string modelId;            // <= 32 chars
  std::string configFingerprint;  // <= 16 chars
  uint64_t chunkId = 0;
  uint64_t runSeed = 0;
  uint16_t moveCap = 0;
  uint16_t temperatureMoves = 0;
  uint8_t recordExtras = 0;
  uint16_t reducedSimulations = 0;
  float fullSearchProb = 1.0f;
};

struct MoveVisits {
  uint32_t rootTotalVisits = 0;
  std::vector<std::pair<uint16_t, uint32_t>> counts;  // (action, count), action n*n = pass
};

struct GameRecord {
  uint16_t gameIndex = 0;
  int8_t result = 0;  // +1 black, -1 white, 0 draw
  Termination termination = Termination::TwoPasses;
  bool noResignGame = false;
  float score = 0.0f;  // black - white - komi (area)
  uint64_t gameSeed = 0;
  std::vector<std::vector<uint8_t>> snapshots;  // T+1 packed snapshots, P = ceil(n*n/4) bytes each
  std::vector<uint16_t> moves;                  // T entries, n*n = pass
  std::vector<uint8_t> searchKind;              // T entries if kExtrasSearchKind, else empty
  std::vector<float> rootValue;                 // T
  std::vector<float> rootMaxQ;                  // T
  std::vector<MoveVisits> visits;               // T
  std::vector<int8_t> finalOwnership;           // n*n if kExtrasFinalOwnership, else empty

  int T() const { return static_cast<int>(moves.size()); }
};

struct Chunk {
  ChunkHeader header;
  std::vector<GameRecord> games;
};

inline int packedSnapshotBytes(int n) { return (n * n + 3) / 4; }
// 2 bits per point: 0 empty, 1 black, 2 white; point 4k in bits 0-1 of byte k.
void packSnapshot(const Color* cellsByPoint, int n, uint8_t* out);
inline Color unpackPoint(const uint8_t* packed, int p) {
  const int v = (packed[p >> 2] >> (2 * (p & 3))) & 3;
  return v == 1 ? Color::Black : (v == 2 ? Color::White : Color::Empty);
}

// In-memory serialisation (used by the file writer and by tests).
void serializeHeader(const ChunkHeader& h, std::vector<uint8_t>& out);  // exactly 128 bytes
void serializeGame(const GameRecord& g, const ChunkHeader& h, std::vector<uint8_t>& out);
// Throws std::runtime_error on malformed input.
Chunk parseChunk(const std::vector<uint8_t>& bytes);

// Validates a record against the header (sizes, sum of counts, action ranges); returns
// an empty string if fine.
std::string validateGame(const GameRecord& g, const ChunkHeader& h);

// Streaming writer with atomic publish: writes "<path>.tmp", rewrites the header with
// the final counts on close() and renames to <path>. Nothing is visible until then.
class ChunkWriter {
 public:
  ChunkWriter(const std::string& path, const ChunkHeader& header);
  ~ChunkWriter();
  void append(const GameRecord& g);  // validates; assigns g.gameIndex = current count
  int games() const { return static_cast<int>(header_.numGames); }
  // Finalises and publishes; returns the final path. Safe to call once.
  std::string close();
  // Discards the temporary file (no publish).
  void discard();

 private:
  std::string path_;
  ChunkHeader header_;
  std::unique_ptr<std::ofstream> out_;
  std::vector<uint8_t> scratch_;
  bool open_ = true;
};

Chunk readChunk(const std::string& path);
void writeChunk(const std::string& path, const Chunk& chunk);  // via ChunkWriter

}  // namespace mango
