#include "selfplay/chunk.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace mango {

namespace {

// --- little-endian encoders -----------------------------------------------------------
void putU8(std::vector<uint8_t>& o, uint8_t v) { o.push_back(v); }
void putI8(std::vector<uint8_t>& o, int8_t v) { o.push_back(static_cast<uint8_t>(v)); }
void putU16(std::vector<uint8_t>& o, uint16_t v) {
  o.push_back(static_cast<uint8_t>(v & 0xFF));
  o.push_back(static_cast<uint8_t>(v >> 8));
}
void putU32(std::vector<uint8_t>& o, uint32_t v) {
  for (int i = 0; i < 4; ++i) o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void putU64(std::vector<uint8_t>& o, uint64_t v) {
  for (int i = 0; i < 8; ++i) o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void putF32(std::vector<uint8_t>& o, float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);  // IEEE-754 bit pattern, then little-endian bytes
  putU32(o, u);
}
void putStr(std::vector<uint8_t>& o, const std::string& s, size_t width) {
  if (s.size() > width) throw std::runtime_error("string field too long: " + s);
  for (size_t i = 0; i < width; ++i) o.push_back(i < s.size() ? static_cast<uint8_t>(s[i]) : 0);
}
void putZeros(std::vector<uint8_t>& o, size_t n) { o.insert(o.end(), n, 0); }

// --- reader with bounds checks ---------------------------------------------------------
struct Cursor {
  const std::vector<uint8_t>& b;
  size_t pos = 0;
  void need(size_t n) const {
    if (pos + n > b.size()) throw std::runtime_error("chunk truncated at byte " + std::to_string(pos));
  }
  uint8_t u8() {
    need(1);
    return b[pos++];
  }
  int8_t i8() { return static_cast<int8_t>(u8()); }
  uint16_t u16() {
    need(2);
    uint16_t v = static_cast<uint16_t>(b[pos] | (b[pos + 1] << 8));
    pos += 2;
    return v;
  }
  uint32_t u32() {
    need(4);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[pos + i]) << (8 * i);
    pos += 4;
    return v;
  }
  uint64_t u64() {
    need(8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[pos + i]) << (8 * i);
    pos += 8;
    return v;
  }
  float f32() {
    uint32_t u = u32();
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  std::string str(size_t width) {
    need(width);
    std::string s;
    for (size_t i = 0; i < width; ++i) {
      if (b[pos + i] == 0) break;
      s.push_back(static_cast<char>(b[pos + i]));
    }
    pos += width;
    return s;
  }
  void skip(size_t n) {
    need(n);
    pos += n;
  }
  void bytes(uint8_t* dst, size_t n) {
    need(n);
    std::memcpy(dst, b.data() + pos, n);
    pos += n;
  }
};

}  // namespace

void packSnapshot(const Color* cells, int n, uint8_t* out) {
  const int nn = n * n;
  const int bytes = packedSnapshotBytes(n);
  std::memset(out, 0, bytes);
  for (int p = 0; p < nn; ++p) {
    const int v = cells[p] == Color::Black ? 1 : (cells[p] == Color::White ? 2 : 0);
    out[p >> 2] |= static_cast<uint8_t>(v << (2 * (p & 3)));
  }
}

void serializeHeader(const ChunkHeader& h, std::vector<uint8_t>& out) {
  out.clear();
  out.reserve(kChunkHeaderSize);
  out.insert(out.end(), {'M', 'G', 'O', '2'});           // 0
  putU8(out, kChunkVersion);                              // 4
  putU8(out, static_cast<uint8_t>(h.boardSize));          // 5
  putU8(out, h.planes);                                   // 6
  putU8(out, h.featureSchema);                            // 7
  putU8(out, h.rulesId);                                  // 8
  putZeros(out, 3);                                       // 9
  putF32(out, h.komi);                                    // 12
  putU32(out, kChunkHeaderSize);                          // 16
  putU32(out, h.numGames);                                // 20
  putU32(out, h.numPositions);                            // 24
  putU32(out, h.simulationsPerMove);                      // 28
  putStr(out, h.modelId, 32);                             // 32
  putStr(out, h.configFingerprint, 16);                   // 64
  putU64(out, h.chunkId);                                 // 80
  putU64(out, h.runSeed);                                 // 88
  putU16(out, h.moveCap);                                 // 96
  putU16(out, h.temperatureMoves);                        // 98
  putU8(out, h.recordExtras);                             // 100
  putU8(out, 0);                                          // 101
  putU16(out, h.reducedSimulations);                      // 102
  putF32(out, h.fullSearchProb);                          // 104
  putZeros(out, 20);                                      // 108
  if (out.size() != kChunkHeaderSize) throw std::logic_error("header size mismatch");
}

std::string validateGame(const GameRecord& g, const ChunkHeader& h) {
  const int n = h.boardSize;
  const int nn = n * n;
  const int T = g.T();
  if (T > 65535) return "T exceeds u16";
  if (static_cast<int>(g.snapshots.size()) != T + 1) return "snapshots.size() != T + 1";
  for (const auto& s : g.snapshots)
    if (static_cast<int>(s.size()) != packedSnapshotBytes(n)) return "snapshot has the wrong byte count";
  if (static_cast<int>(g.rootValue.size()) != T || static_cast<int>(g.rootMaxQ.size()) != T) return "root value arrays != T";
  if (static_cast<int>(g.visits.size()) != T) return "visits.size() != T";
  for (uint16_t m : g.moves)
    if (m > nn) return "move out of range";
  for (int t = 0; t < T; ++t) {
    const MoveVisits& v = g.visits[t];
    if (v.counts.empty()) return "move " + std::to_string(t) + " has no visit entries";
    uint64_t sum = 0;
    for (const auto& [a, c] : v.counts) {
      if (a > nn) return "visit action out of range";
      if (c == 0) return "zero visit count stored";
      sum += c;
    }
    const bool pruned = (h.recordExtras & kExtrasPrunedVisits) != 0;
    if (!pruned && sum != v.rootTotalVisits) return "sum of counts != root_total_visits at move " + std::to_string(t);
    if (pruned && sum > v.rootTotalVisits) return "sum of pruned counts > root_total_visits at move " + std::to_string(t);
  }
  const bool wantKind = (h.recordExtras & kExtrasSearchKind) != 0;
  if (wantKind && static_cast<int>(g.searchKind.size()) != T) return "search_kind.size() != T";
  if (!wantKind && !g.searchKind.empty()) return "search_kind present but header bit 1 is clear";
  const bool wantOwn = (h.recordExtras & kExtrasFinalOwnership) != 0;
  if (wantOwn && static_cast<int>(g.finalOwnership.size()) != nn) return "final_ownership.size() != n*n";
  if (!wantOwn && !g.finalOwnership.empty()) return "final_ownership present but header bit 0 is clear";
  if (g.result < -1 || g.result > 1) return "result out of range";
  return "";
}

void serializeGame(const GameRecord& g, const ChunkHeader& h, std::vector<uint8_t>& out) {
  const std::string err = validateGame(g, h);
  if (!err.empty()) throw std::runtime_error("invalid game record: " + err);
  const int T = g.T();
  putU16(out, g.gameIndex);                                  // 0
  putU16(out, static_cast<uint16_t>(T));                     // 2
  putI8(out, g.result);                                      // 4
  putU8(out, static_cast<uint8_t>(g.termination));           // 5
  putU8(out, g.noResignGame ? 1 : 0);                        // 6
  putU8(out, 0);                                             // 7
  putF32(out, g.score);                                      // 8
  putU64(out, g.gameSeed);                                   // 12
  for (const auto& s : g.snapshots) out.insert(out.end(), s.begin(), s.end());  // 20
  for (uint16_t m : g.moves) putU16(out, m);
  if (h.recordExtras & kExtrasSearchKind)
    for (uint8_t k : g.searchKind) putU8(out, k);
  for (int t = 0; t < T; ++t) {
    putF32(out, g.rootValue[t]);
    putF32(out, g.rootMaxQ[t]);
  }
  for (int t = 0; t < T; ++t) {
    const MoveVisits& v = g.visits[t];
    putU32(out, v.rootTotalVisits);
    putU16(out, static_cast<uint16_t>(v.counts.size()));
    for (const auto& [a, c] : v.counts) {
      putU16(out, a);
      putU32(out, c);
    }
  }
  if (h.recordExtras & kExtrasFinalOwnership)
    for (int8_t o : g.finalOwnership) putI8(out, o);
}

Chunk parseChunk(const std::vector<uint8_t>& bytes) {
  Cursor c{bytes};
  Chunk chunk;
  ChunkHeader& h = chunk.header;
  c.need(kChunkHeaderSize);
  if (bytes[0] != 'M' || bytes[1] != 'G' || bytes[2] != 'O' || bytes[3] != '2') throw std::runtime_error("bad magic");
  c.pos = 4;
  const uint8_t version = c.u8();
  if (version != kChunkVersion) throw std::runtime_error("unsupported chunk version " + std::to_string(version));
  h.boardSize = c.u8();
  h.planes = c.u8();
  h.featureSchema = c.u8();
  h.rulesId = c.u8();
  c.skip(3);
  h.komi = c.f32();
  const uint32_t headerSize = c.u32();
  if (headerSize != kChunkHeaderSize) throw std::runtime_error("header_size != 128");
  h.numGames = c.u32();
  h.numPositions = c.u32();
  h.simulationsPerMove = c.u32();
  h.modelId = c.str(32);
  h.configFingerprint = c.str(16);
  h.chunkId = c.u64();
  h.runSeed = c.u64();
  h.moveCap = c.u16();
  h.temperatureMoves = c.u16();
  h.recordExtras = c.u8();
  c.skip(1);
  h.reducedSimulations = c.u16();
  h.fullSearchProb = c.f32();
  c.skip(20);
  if (h.boardSize < kMinN || h.boardSize > kMaxN) throw std::runtime_error("bad board size in header");
  const int n = h.boardSize;
  const int nn = n * n;
  const int P = packedSnapshotBytes(n);
  uint64_t positions = 0;
  for (uint32_t gi = 0; gi < h.numGames; ++gi) {
    GameRecord g;
    g.gameIndex = c.u16();
    const int T = c.u16();
    g.result = c.i8();
    g.termination = static_cast<Termination>(c.u8());
    g.noResignGame = c.u8() != 0;
    c.skip(1);
    g.score = c.f32();
    g.gameSeed = c.u64();
    g.snapshots.resize(T + 1);
    for (auto& s : g.snapshots) {
      s.resize(P);
      c.bytes(s.data(), P);
    }
    g.moves.resize(T);
    for (auto& m : g.moves) m = c.u16();
    if (h.recordExtras & kExtrasSearchKind) {
      g.searchKind.resize(T);
      for (auto& k : g.searchKind) k = c.u8();
    }
    g.rootValue.resize(T);
    g.rootMaxQ.resize(T);
    for (int t = 0; t < T; ++t) {
      g.rootValue[t] = c.f32();
      g.rootMaxQ[t] = c.f32();
    }
    g.visits.resize(T);
    for (int t = 0; t < T; ++t) {
      MoveVisits& v = g.visits[t];
      v.rootTotalVisits = c.u32();
      const int nnz = c.u16();
      v.counts.resize(nnz);
      for (auto& [a, cnt] : v.counts) {
        a = c.u16();
        cnt = c.u32();
      }
    }
    if (h.recordExtras & kExtrasFinalOwnership) {
      g.finalOwnership.resize(nn);
      for (auto& o : g.finalOwnership) o = c.i8();
    }
    const std::string err = validateGame(g, h);
    if (!err.empty()) throw std::runtime_error("game " + std::to_string(gi) + ": " + err);
    positions += T;
    chunk.games.push_back(std::move(g));
  }
  if (c.pos != bytes.size()) throw std::runtime_error("trailing bytes after the last game");
  if (positions != h.numPositions) throw std::runtime_error("num_positions does not match the records");
  return chunk;
}

// --- file writer -----------------------------------------------------------------------

ChunkWriter::ChunkWriter(const std::string& path, const ChunkHeader& header) : path_(path), header_(header) {
  header_.numGames = 0;
  header_.numPositions = 0;
  out_ = std::make_unique<std::ofstream>(path_ + ".tmp", std::ios::binary | std::ios::trunc);
  if (!*out_) throw std::runtime_error("cannot create " + path_ + ".tmp");
  serializeHeader(header_, scratch_);
  out_->write(reinterpret_cast<const char*>(scratch_.data()), static_cast<std::streamsize>(scratch_.size()));
}

ChunkWriter::~ChunkWriter() {
  if (open_) discard();
}

void ChunkWriter::append(const GameRecord& g) {
  if (!open_) throw std::logic_error("append on a closed ChunkWriter");
  GameRecord copy = g;
  copy.gameIndex = static_cast<uint16_t>(header_.numGames);
  scratch_.clear();
  serializeGame(copy, header_, scratch_);
  out_->write(reinterpret_cast<const char*>(scratch_.data()), static_cast<std::streamsize>(scratch_.size()));
  if (!*out_) throw std::runtime_error("write failed: " + path_ + ".tmp");
  header_.numGames += 1;
  header_.numPositions += static_cast<uint32_t>(g.T());
}

std::string ChunkWriter::close() {
  if (!open_) return path_;
  out_->seekp(0);
  serializeHeader(header_, scratch_);
  out_->write(reinterpret_cast<const char*>(scratch_.data()), static_cast<std::streamsize>(scratch_.size()));
  out_->flush();
  if (!*out_) throw std::runtime_error("write failed: " + path_ + ".tmp");
  out_.reset();
  open_ = false;
  std::filesystem::rename(path_ + ".tmp", path_);
  return path_;
}

void ChunkWriter::discard() {
  if (!open_) return;
  out_.reset();
  open_ = false;
  std::error_code ec;
  std::filesystem::remove(path_ + ".tmp", ec);
}

Chunk readChunk(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parseChunk(bytes);
}

void writeChunk(const std::string& path, const Chunk& chunk) {
  ChunkWriter w(path, chunk.header);
  for (const GameRecord& g : chunk.games) w.append(g);
  w.close();
}

}  // namespace mango
