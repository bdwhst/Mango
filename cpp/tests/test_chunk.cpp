// MGO2 chunk format tests (DESIGN 5.6, 9 "Chunk format"): byte-exact fixtures shared
// with python/tests/test_chunk.py, round trips, validation, atomic publish, and the
// cross-language feature-parity fixture.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/board.h"
#include "core/features.h"
#include "core/symmetry.h"
#include "doctest/doctest.h"
#include "nlohmann/json.hpp"
#include "selfplay/chunk.h"
#include "tests/test_util.h"

using namespace mango;
using namespace mango::test;

#ifndef MANGO_FIXTURE_DIR
#define MANGO_FIXTURE_DIR "cpp/tests/fixtures"
#endif

namespace {

const std::string kFixtureDir = MANGO_FIXTURE_DIR;
namespace fs = std::filesystem;

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// Compares `bytes` with the committed fixture; writes the fixture when
// MANGO_WRITE_FIXTURES=1 and the file is missing (generation step, never in CI).
void checkFixture(const std::string& name, const std::vector<uint8_t>& bytes) {
  const std::string path = kFixtureDir + "/" + name;
  if (!fs::exists(path)) {
    const char* w = std::getenv("MANGO_WRITE_FIXTURES");
    const bool mayWrite = w != nullptr && std::string(w) == "1";
    REQUIRE_MESSAGE(mayWrite, "missing fixture " << path << " (set MANGO_WRITE_FIXTURES=1 to generate)");
    writeFile(path, bytes);
  }
  const std::vector<uint8_t> committed = readFile(path);
  REQUIRE(committed.size() == bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (committed[i] != bytes[i]) {
      FAIL("fixture " << name << " differs at byte " << i);
    }
  }
}

void packBoard(const Board& b, std::vector<uint8_t>& out) {
  const int n = b.size();
  std::vector<Color> cells(n * n);
  for (int p = 0; p < n * n; ++p) cells[p] = b.atPoint(p);
  out.resize(packedSnapshotBytes(n));
  packSnapshot(cells.data(), n, out.data());
}

// Scripted 5x5 game with hand-chosen statistics; `extras` selects the optional fields.
GameRecord scriptedGame(const std::vector<Move>& moves, uint64_t seed, uint8_t extras, bool pruned) {
  const int n = 5, nn = 25;
  Board b(n, 7.5f);
  GameRecord g;
  g.gameSeed = seed;
  g.snapshots.emplace_back();
  packBoard(b, g.snapshots.back());
  for (size_t t = 0; t < moves.size(); ++t) {
    const Move m = moves[t];
    g.moves.push_back(static_cast<uint16_t>(m == kPass ? nn : m));
    g.rootValue.push_back(0.25f - 0.1f * static_cast<float>(t));
    g.rootMaxQ.push_back(-0.5f + 0.2f * static_cast<float>(t));
    MoveVisits v;
    v.counts.emplace_back(static_cast<uint16_t>(m == kPass ? nn : m), 3);
    v.counts.emplace_back(static_cast<uint16_t>((m == kPass ? 0 : (m + 1) % nn)), 1);
    v.rootTotalVisits = pruned ? 5 : 4;
    g.visits.push_back(v);
    if (extras & kExtrasSearchKind) g.searchKind.push_back(t == 1 ? 0 : 1);
    b.play(m);
    g.snapshots.emplace_back();
    packBoard(b, g.snapshots.back());
  }
  g.score = b.score();
  g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
  g.termination = b.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
  if (extras & kExtrasFinalOwnership) {
    g.finalOwnership.resize(nn);
    b.areaOwnership(g.finalOwnership.data());
  }
  return g;
}

Chunk scriptedChunk(uint8_t extras) {
  Chunk c;
  ChunkHeader& h = c.header;
  h.boardSize = 5;
  h.komi = 7.5f;
  h.simulationsPerMove = 4;
  h.modelId = "0001-deadbeef";
  h.configFingerprint = "0123456789abcdef";
  h.chunkId = 42;
  h.runSeed = 7;
  h.moveCap = 50;
  h.temperatureMoves = 4;
  h.recordExtras = extras;
  h.reducedSimulations = (extras & kExtrasSearchKind) ? 2 : 0;
  h.fullSearchProb = (extras & kExtrasSearchKind) ? 0.25f : 1.0f;
  const bool pruned = (extras & kExtrasPrunedVisits) != 0;
  // Game 0: centre, opposite corner-ish, two passes. Game 1: black captures a white stone.
  auto P = [](int r, int c) { return static_cast<Move>(pointOf(r, c, 5)); };
  GameRecord g0 = scriptedGame({P(2, 2), P(1, 1), kPass, kPass}, 1001, extras, pruned);
  GameRecord g1 = scriptedGame({P(0, 1), P(0, 0), P(1, 0), kPass, kPass}, 1002, extras, pruned);
  g0.gameIndex = 0;
  g1.gameIndex = 1;
  g1.noResignGame = true;
  c.games = {g0, g1};
  h.numGames = 2;
  h.numPositions = 9;
  return c;
}

std::vector<uint8_t> serializeChunk(const Chunk& c) {
  std::vector<uint8_t> bytes;
  serializeHeader(c.header, bytes);
  for (const GameRecord& g : c.games) serializeGame(g, c.header, bytes);
  return bytes;
}

bool sameGame(const GameRecord& a, const GameRecord& b) {
  if (a.gameIndex != b.gameIndex || a.result != b.result || a.termination != b.termination ||
      a.noResignGame != b.noResignGame || a.score != b.score || a.gameSeed != b.gameSeed)
    return false;
  if (a.snapshots != b.snapshots || a.moves != b.moves || a.searchKind != b.searchKind) return false;
  if (a.rootValue != b.rootValue || a.rootMaxQ != b.rootMaxQ || a.finalOwnership != b.finalOwnership) return false;
  if (a.visits.size() != b.visits.size()) return false;
  for (size_t t = 0; t < a.visits.size(); ++t)
    if (a.visits[t].rootTotalVisits != b.visits[t].rootTotalVisits || a.visits[t].counts != b.visits[t].counts) return false;
  return true;
}

}  // namespace

TEST_CASE("chunk header is 128 bytes with the documented offsets") {
  Chunk c = scriptedChunk(0);
  std::vector<uint8_t> h;
  serializeHeader(c.header, h);
  REQUIRE(h.size() == 128);
  CHECK(std::string(h.begin(), h.begin() + 4) == "MGO2");
  CHECK(h[4] == 2);      // version
  CHECK(h[5] == 5);      // board size
  CHECK(h[6] == 17);     // planes
  CHECK(h[7] == 1);      // feature schema
  CHECK(h[8] == 1);      // rules id
  CHECK((h[9] | h[10] | h[11]) == 0);
  CHECK(h[12] == 0x00);  // komi 7.5f = 0x40F00000 little-endian
  CHECK(h[13] == 0x00);
  CHECK(h[14] == 0xF0);
  CHECK(h[15] == 0x40);
  CHECK(h[16] == 128);   // header_size
  CHECK(h[20] == 2);     // num_games
  CHECK(h[24] == 9);     // num_positions
  CHECK(h[28] == 4);     // simulations
  CHECK(std::string(h.begin() + 32, h.begin() + 45) == "0001-deadbeef");
  CHECK(h[45] == 0);     // zero padded
  CHECK(std::string(h.begin() + 64, h.begin() + 80) == "0123456789abcdef");
  CHECK(h[80] == 42);    // chunk id
  CHECK(h[88] == 7);     // run seed
  CHECK(h[96] == 50);    // move cap
  CHECK(h[98] == 4);     // temperature moves
  CHECK(h[100] == 0);    // record_extras
  CHECK(h[102] == 0);    // reduced simulations
  CHECK(h[104] == 0x00); // full_search_prob 1.0f = 0x3F800000
  CHECK(h[106] == 0x80);
  CHECK(h[107] == 0x3F);
  for (int i = 108; i < 128; ++i) CHECK(h[i] == 0);
  // A too-long model id is refused rather than truncated silently.
  c.header.modelId = std::string(33, 'x');
  CHECK_THROWS(serializeHeader(c.header, h));
}

TEST_CASE("snapshot packing: 2 bits per point, little-endian within a byte") {
  const int n = 5;
  Board b = Board::fromString(
      "X O . . .\n"
      ". . . . .\n"
      ". . . . .\n"
      ". . . . .\n"
      ". . . . O\n",
      7.5f);
  std::vector<uint8_t> packed;
  packBoard(b, packed);
  REQUIRE(static_cast<int>(packed.size()) == 7);  // ceil(25/4)
  CHECK(packed[0] == (1 | (2 << 2)));             // point 0 black, point 1 white
  CHECK(packed[6] == 2);                          // point 24 white in bits 0-1 of byte 6
  for (int p = 0; p < n * n; ++p) CHECK(unpackPoint(packed.data(), p) == b.atPoint(p));
}

TEST_CASE("chunk fixtures are byte-exact and parse back to the scripted games") {
  for (uint8_t extras : {uint8_t{0}, uint8_t{kExtrasFinalOwnership | kExtrasSearchKind | kExtrasPrunedVisits}}) {
    Chunk c = scriptedChunk(extras);
    std::vector<uint8_t> bytes = serializeChunk(c);
    checkFixture(extras == 0 ? "mgo2_fixture.bin" : "mgo2_fixture_extras.bin", bytes);
    Chunk back = parseChunk(bytes);
    CHECK(back.header.boardSize == 5);
    CHECK(back.header.modelId == "0001-deadbeef");
    CHECK(back.header.configFingerprint == "0123456789abcdef");
    CHECK(back.header.chunkId == 42);
    CHECK(back.header.recordExtras == extras);
    CHECK(back.header.numGames == 2);
    CHECK(back.header.numPositions == 9);
    REQUIRE(back.games.size() == 2);
    CHECK(sameGame(back.games[0], c.games[0]));
    CHECK(sameGame(back.games[1], c.games[1]));
    // Known content: game 1 captures the white stone at (0,0) with move 2.
    const GameRecord& g1 = back.games[1];
    CHECK(unpackPoint(g1.snapshots[2].data(), pointOf(0, 0, 5)) == Color::White);
    CHECK(unpackPoint(g1.snapshots[3].data(), pointOf(0, 0, 5)) == Color::Empty);
    CHECK(g1.moves[3] == 25);  // pass
    CHECK(g1.termination == Termination::TwoPasses);
    CHECK(g1.result == 1);  // black owns the whole board: 25 - 0 - 7.5
    CHECK(g1.score == doctest::Approx(17.5f));
    CHECK(g1.noResignGame);
    if (extras) {
      CHECK(g1.searchKind == std::vector<uint8_t>{1, 0, 1, 1, 1});
      CHECK(static_cast<int>(g1.finalOwnership.size()) == 25);
      CHECK(g1.finalOwnership[pointOf(0, 0, 5)] == 1);
      CHECK(g1.visits[0].rootTotalVisits == 5);  // pruned: 3 + 1 <= 5
    }
  }
}

TEST_CASE("chunk validation rejects inconsistent records") {
  Chunk c = scriptedChunk(0);
  GameRecord g = c.games[0];
  CHECK(validateGame(g, c.header) == "");
  GameRecord bad = g;
  bad.visits[0].rootTotalVisits = 9;
  CHECK(validateGame(bad, c.header) != "");  // sum 4 != 9 without the pruned bit
  ChunkHeader prunedHeader = c.header;
  prunedHeader.recordExtras = kExtrasPrunedVisits;
  CHECK(validateGame(bad, prunedHeader) == "");  // sum 4 <= 9 is fine when pruned
  bad = g;
  bad.moves[0] = 26;
  CHECK(validateGame(bad, c.header) != "");
  bad = g;
  bad.snapshots.pop_back();
  CHECK(validateGame(bad, c.header) != "");
  bad = g;
  bad.finalOwnership.assign(25, 0);
  CHECK(validateGame(bad, c.header) != "");  // present but header bit clear
  std::vector<uint8_t> bytes = serializeChunk(c);
  std::vector<uint8_t> truncated(bytes.begin(), bytes.end() - 3);
  CHECK_THROWS(parseChunk(truncated));
  std::vector<uint8_t> extra = bytes;
  extra.push_back(0);
  CHECK_THROWS(parseChunk(extra));
  bytes[0] = 'X';
  CHECK_THROWS(parseChunk(bytes));
}

TEST_CASE("chunk writer publishes atomically and round-trips random games") {
  const fs::path dir = fs::temp_directory_path() / "mango_test_chunk";
  fs::create_directories(dir);
  const std::string path = (dir / "chunk_000001.mgo").string();
  std::error_code ec;
  fs::remove(path, ec);
  fs::remove(path + ".tmp", ec);

  Chunk c;
  c.header.boardSize = 7;
  c.header.komi = 7.5f;
  c.header.modelId = "0002-abcdef01";
  c.header.moveCap = 98;
  c.header.recordExtras = kExtrasFinalOwnership | kExtrasSearchKind;
  const int n = 7, nn = 49;
  for (int gi = 0; gi < 5; ++gi) {
    GameRecord g;
    g.gameSeed = 100 + gi;
    Rng rng(g.gameSeed);
    Board last(n, 7.5f);
    randomGame(n, 7.5f, 500 + gi, 0.05, [&](const Board& b, const RefBoard&, const GameHistory& h) {
      g.snapshots.emplace_back();
      packBoard(b, g.snapshots.back());
      last = b;
      if (b.gameOver()) return;
      (void)h;
      MoveVisits v;
      const uint32_t total = 10 + rng.uniformInt(20);
      uint32_t left = total;
      std::vector<Move> legal;
      b.legalMoves(HashHistory(h), legal);
      for (size_t i = 0; i < legal.size() && left > 0; ++i) {
        const uint32_t cnt = (i + 1 == legal.size()) ? left : 1 + rng.uniformInt(left);
        v.counts.emplace_back(static_cast<uint16_t>(legal[i] == kPass ? nn : legal[i]), cnt);
        left -= cnt;
        if (cnt == 0) v.counts.pop_back();
      }
      v.rootTotalVisits = total;
      g.visits.push_back(v);
      g.rootValue.push_back(static_cast<float>(rng.uniformReal() * 2 - 1));
      g.rootMaxQ.push_back(static_cast<float>(rng.uniformReal() * 2 - 1));
      g.searchKind.push_back(1);
    });
    // Moves are the difference between consecutive snapshots: replay them from the history.
    // randomGame does not expose the moves, so record them via the board's lastMove.
    (void)last;
    g.moves.clear();
    // Reconstruct moves by replaying: for each t, find the point that became occupied,
    // or pass. Captures never add stones, so a new stone identifies the move uniquely.
    for (size_t t = 0; t + 1 < g.snapshots.size(); ++t) {
      int placed = -1;
      for (int p = 0; p < nn; ++p)
        if (unpackPoint(g.snapshots[t].data(), p) == Color::Empty && unpackPoint(g.snapshots[t + 1].data(), p) != Color::Empty)
          placed = p;
      g.moves.push_back(static_cast<uint16_t>(placed < 0 ? nn : placed));
    }
    g.score = last.score();
    g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
    g.termination = last.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
    g.finalOwnership.resize(nn);
    last.areaOwnership(g.finalOwnership.data());
    REQUIRE(validateGame(g, c.header) == "");
    c.games.push_back(g);
  }

  {
    ChunkWriter w(path, c.header);
    for (const GameRecord& g : c.games) w.append(g);
    CHECK(fs::exists(path + ".tmp"));
    CHECK_FALSE(fs::exists(path));
    CHECK(w.games() == 5);
    CHECK(w.close() == path);
  }
  CHECK(fs::exists(path));
  CHECK_FALSE(fs::exists(path + ".tmp"));
  Chunk back = readChunk(path);
  CHECK(back.header.numGames == 5);
  uint32_t positions = 0;
  for (const GameRecord& g : c.games) positions += static_cast<uint32_t>(g.T());
  CHECK(back.header.numPositions == positions);
  REQUIRE(back.games.size() == 5);
  for (size_t i = 0; i < 5; ++i) {
    GameRecord expected = c.games[i];
    expected.gameIndex = static_cast<uint16_t>(i);  // assigned by the writer
    CHECK(sameGame(back.games[i], expected));
  }
  // discard() leaves nothing behind.
  {
    ChunkWriter w(path + ".b", c.header);
    w.append(c.games[0]);
    w.discard();
  }
  CHECK_FALSE(fs::exists(path + ".b"));
  CHECK_FALSE(fs::exists(path + ".b.tmp"));
  // A writer destroyed without close() does not publish.
  { ChunkWriter w(path + ".c", c.header); }
  CHECK_FALSE(fs::exists(path + ".c"));
  CHECK_FALSE(fs::exists(path + ".c.tmp"));
}

TEST_CASE("feature and symmetry parity fixtures for the Python loader") {
  // A random 5x5 game as a chunk plus the C++ feature planes at every t < T; the Python
  // loader must assemble identical planes from the snapshots (python/tests/test_feature_parity.py).
  const int n = 5, nn = 25;
  Chunk c;
  c.header.boardSize = n;
  c.header.komi = 7.5f;
  c.header.modelId = "parity";
  c.header.moveCap = 50;
  GameRecord g;
  g.gameSeed = 20;  // % 20 == 0: a holdout game, so the loader test covers the tag too
  std::vector<uint8_t> features;
  Board last(n, 7.5f);
  randomGame(n, 7.5f, 77, 0.08, [&](const Board& b, const RefBoard&, const GameHistory&) {
    g.snapshots.emplace_back();
    packBoard(b, g.snapshots.back());
    last = b;
    if (b.gameOver()) return;
    std::vector<uint8_t> planes(featureSize(n));
    encodeFeatures(b, planes.data());
    features.insert(features.end(), planes.begin(), planes.end());
    MoveVisits v;
    v.counts.emplace_back(static_cast<uint16_t>(b.moveCount() % nn), 2);
    v.counts.emplace_back(static_cast<uint16_t>(nn), 1);  // pass gets a visit at every move
    v.rootTotalVisits = 3;
    g.visits.push_back(v);
    g.rootValue.push_back(0.5f);
    g.rootMaxQ.push_back(0.25f);
  });
  for (size_t t = 0; t + 1 < g.snapshots.size(); ++t) {
    int placed = -1;
    for (int p = 0; p < nn; ++p)
      if (unpackPoint(g.snapshots[t].data(), p) == Color::Empty && unpackPoint(g.snapshots[t + 1].data(), p) != Color::Empty)
        placed = p;
    g.moves.push_back(static_cast<uint16_t>(placed < 0 ? nn : placed));
  }
  g.score = last.score();
  g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
  g.termination = last.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
  REQUIRE(validateGame(g, c.header) == "");
  REQUIRE(g.T() >= 8);  // long enough to exercise the 8-step history
  c.games.push_back(g);
  c.header.numGames = 1;
  c.header.numPositions = static_cast<uint32_t>(g.T());
  checkFixture("parity_5x5.mgo", serializeChunk(c));
  checkFixture("parity_5x5.features", features);

  // Symmetry tables: transformPoint for every symmetry, n = 5, as JSON.
  nlohmann::json tables = nlohmann::json::array();
  for (int sym = 0; sym < kNumSymmetries; ++sym) {
    nlohmann::json row = nlohmann::json::array();
    for (int p = 0; p < nn; ++p) row.push_back(transformPoint(p, n, sym));
    tables.push_back(row);
  }
  nlohmann::json j;
  j["board_size"] = n;
  j["transform_point"] = tables;
  nlohmann::json inv = nlohmann::json::array();
  for (int s = 0; s < kNumSymmetries; ++s) inv.push_back(inverseSymmetry(s));
  j["inverse"] = inv;
  const std::string text = j.dump();
  checkFixture("symmetry_5x5.json", std::vector<uint8_t>(text.begin(), text.end()));
}
