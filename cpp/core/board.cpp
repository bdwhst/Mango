#include "core/board.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

#include "core/symmetry.h"
#include "core/zobrist.h"

namespace mango {

namespace {
const Snapshot kZeroSnapshot{};
}

Board::Board(int n, float komi, int moveCap)
    : n_(n), komi_(komi), moveCap_(moveCap < 0 ? 2 * n * n : moveCap) {
  if (n < kMinN || n > kMaxN) throw std::invalid_argument("board size out of range");
  for (int i = 0; i < kMaxCells; ++i) {
    cells_[i] = Color::Wall;
    chainId_[i] = -1;
    next_[i] = -1;
    chainSize_[i] = 0;
  }
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) cells_[locOf(r, c)] = Color::Empty;
  hash_ = 0;
  toMove_ = Color::Black;
  moveCount_ = 0;
  passes_ = 0;
  lastMove_ = kNoMove;
  snapHead_ = 0;
  snapCount_ = 0;
  pushSnapshot();
}

Board Board::fromString(const std::string& diagram, float komi, Color toMove, int moveCap) {
  std::vector<std::string> rows;
  std::string cur;
  for (char ch : diagram) {
    if (ch == '\n') {
      if (!cur.empty()) rows.push_back(cur);
      cur.clear();
    } else if (ch == '.' || ch == 'X' || ch == 'O') {
      cur.push_back(ch);
    }
  }
  if (!cur.empty()) rows.push_back(cur);
  const int n = static_cast<int>(rows.size());
  Board b(n, komi, moveCap);
  for (int r = 0; r < n; ++r) {
    if (static_cast<int>(rows[r].size()) != n) throw std::invalid_argument("diagram is not square");
    for (int c = 0; c < n; ++c) {
      char ch = rows[r][c];
      b.cells_[locOf(r, c)] = ch == 'X' ? Color::Black : (ch == 'O' ? Color::White : Color::Empty);
    }
  }
  b.rebuildChains();
  b.toMove_ = toMove;
  b.snapHead_ = 0;
  b.snapCount_ = 0;
  b.pushSnapshot();
  return b;
}

// ---------------------------------------------------------------------------
// Snapshots

void Board::fillSnapshot(Snapshot& s) const {
  s.stones[0].reset();
  s.stones[1].reset();
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p) {
    Color c = cells_[locOfPoint(p, n_)];
    if (c == Color::Black) s.stones[0].set(p);
    else if (c == Color::White) s.stones[1].set(p);
  }
}

void Board::pushSnapshot() {
  snapHead_ = (snapHead_ + 1) % kHistoryLen;
  fillSnapshot(snaps_[snapHead_]);
  if (snapCount_ < kHistoryLen) ++snapCount_;
}

const Snapshot& Board::snapshot(int stepsBack) const {
  if (stepsBack < 0 || stepsBack >= snapCount_) return kZeroSnapshot;
  return snaps_[(snapHead_ - stepsBack + kHistoryLen) % kHistoryLen];
}

void Board::setCellsFromSnapshot(const Snapshot& s) {
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p) {
    Color c = Color::Empty;
    if (s.stones[0].test(p)) c = Color::Black;
    else if (s.stones[1].test(p)) c = Color::White;
    cells_[locOfPoint(p, n_)] = c;
  }
}

// ---------------------------------------------------------------------------
// Chains

int Board::mergeChains(int rootA, int rootB) {
  if (rootA == rootB) return rootA;
  int big = rootA, small = rootB;
  if (chainSize_[small] > chainSize_[big]) std::swap(big, small);
  // Relabel the smaller chain.
  int s = small;
  do {
    chainId_[s] = static_cast<int16_t>(big);
    s = next_[s];
  } while (s != small);
  // Splice the circular lists.
  int16_t tmp = next_[big];
  next_[big] = next_[small];
  next_[small] = tmp;
  libs_[big].orWith(libs_[small]);
  chainSize_[big] = static_cast<int16_t>(chainSize_[big] + chainSize_[small]);
  chainSize_[small] = 0;
  return big;
}

void Board::removeChain(int root) {
  const Color c = cells_[root];
  const Color enemy = opposite(c);
  // Collect first: the list is destroyed while removing.
  int stones[kMaxPoints];
  int count = 0;
  int s = root;
  do {
    stones[count++] = s;
    s = next_[s];
  } while (s != root);
  for (int i = 0; i < count; ++i) {
    int loc = stones[i];
    cells_[loc] = Color::Empty;
    hash_ ^= zobristKey(c, loc);
    chainId_[loc] = -1;
    next_[loc] = -1;
    chainSize_[loc] = 0;
  }
  // The freed points are liberties of adjacent enemy chains (friendly neighbours were
  // part of this chain, so only enemy chains can be adjacent).
  for (int i = 0; i < count; ++i) {
    int loc = stones[i];
    int p = pointOfLoc(loc, n_);
    for (int d : kDirs) {
      int nb = loc + d;
      if (cells_[nb] == enemy) libs_[chainId_[nb]].set(p);
    }
  }
}

void Board::placeStone(int loc, Color c) {
  const Color enemy = opposite(c);
  const int p = pointOfLoc(loc, n_);
  cells_[loc] = c;
  hash_ ^= zobristKey(c, loc);
  chainId_[loc] = static_cast<int16_t>(loc);
  next_[loc] = static_cast<int16_t>(loc);
  chainSize_[loc] = 1;
  libs_[loc].reset();
  for (int d : kDirs) {
    int nb = loc + d;
    if (cells_[nb] == Color::Empty) libs_[loc].set(pointOfLoc(nb, n_));
  }
  int root = loc;
  for (int d : kDirs) {
    int nb = loc + d;
    if (cells_[nb] == c) root = mergeChains(root, chainId_[nb]);
  }
  libs_[root].clear(p);
  for (int d : kDirs) {
    int nb = loc + d;
    if (cells_[nb] != enemy) continue;
    int er = chainId_[nb];
    libs_[er].clear(p);
    if (!libs_[er].any()) removeChain(er);
  }
}

void Board::rebuildChains() {
  for (int i = 0; i < kMaxCells; ++i) {
    chainId_[i] = -1;
    next_[i] = -1;
    chainSize_[i] = 0;
  }
  hash_ = 0;
  int stack[kMaxPoints];
  int list[kMaxPoints];
  for (int r = 0; r < n_; ++r) {
    for (int c = 0; c < n_; ++c) {
      int loc = locOf(r, c);
      Color col = cells_[loc];
      if (col != Color::Black && col != Color::White) continue;
      hash_ ^= zobristKey(col, loc);
      if (chainId_[loc] != -1) continue;
      int sp = 0, count = 0;
      stack[sp++] = loc;
      chainId_[loc] = static_cast<int16_t>(loc);
      libs_[loc].reset();
      while (sp > 0) {
        int s = stack[--sp];
        list[count++] = s;
        for (int d : kDirs) {
          int nb = s + d;
          if (cells_[nb] == col && chainId_[nb] == -1) {
            chainId_[nb] = static_cast<int16_t>(loc);
            stack[sp++] = nb;
          } else if (cells_[nb] == Color::Empty) {
            libs_[loc].set(pointOfLoc(nb, n_));
          }
        }
      }
      for (int i = 0; i < count; ++i) next_[list[i]] = static_cast<int16_t>(list[(i + 1) % count]);
      chainSize_[loc] = static_cast<int16_t>(count);
    }
  }
}

// ---------------------------------------------------------------------------
// Rules

bool Board::tryMoveHash(Move m, uint64_t* hashAfter) const {
  if (m == kPass) {
    *hashAfter = hash_;
    return true;
  }
  const int loc = locOfPoint(m, n_);
  if (cells_[loc] != Color::Empty) return false;
  const Color c = toMove_;
  const Color enemy = opposite(c);
  uint64_t h = hash_ ^ zobristKey(c, loc);
  bool hasLiberty = false;
  bool captures = false;
  int seen[4];
  int seenCount = 0;
  for (int d : kDirs) {
    int nb = loc + d;
    Color nc = cells_[nb];
    if (nc == Color::Empty) {
      hasLiberty = true;
    } else if (nc == c) {
      if (libs_[chainId_[nb]].count() > 1) hasLiberty = true;  // one liberty is loc itself
    } else if (nc == enemy) {
      int er = chainId_[nb];
      if (libs_[er].count() == 1) {  // its only liberty is loc -> captured
        bool dup = false;
        for (int i = 0; i < seenCount; ++i) dup |= (seen[i] == er);
        if (dup) continue;
        seen[seenCount++] = er;
        captures = true;
        int s = er;
        do {
          h ^= zobristKey(enemy, s);
          s = next_[s];
        } while (s != er);
      }
    }
  }
  if (!hasLiberty && !captures) return false;  // suicide
  *hashAfter = h;
  return true;
}

bool Board::isLegal(Move m, const HashHistory& hist) const {
  if (m == kPass) return true;
  if (m < 0 || m >= n_ * n_) return false;
  uint64_t h;
  if (!tryMoveHash(m, &h)) return false;
  return !hist.contains(h);
}

void Board::legalMoves(const HashHistory& hist, std::vector<Move>& out) const {
  out.clear();
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p)
    if (isLegal(static_cast<Move>(p), hist)) out.push_back(static_cast<Move>(p));
  out.push_back(kPass);
}

void Board::play(Move m) {
  if (m == kPass) {
    ++passes_;
  } else {
    assert(m >= 0 && m < n_ * n_);
    const int loc = locOfPoint(m, n_);
    assert(cells_[loc] == Color::Empty);
    placeStone(loc, toMove_);
    assert(libs_[chainId_[loc]].any() && "suicide is illegal");
    passes_ = 0;
  }
  lastMove_ = m;
  ++moveCount_;
  toMove_ = opposite(toMove_);
  pushSnapshot();
}

// ---------------------------------------------------------------------------
// Scoring

void Board::areaOwnership(int8_t* owner) const {
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p) {
    Color c = cells_[locOfPoint(p, n_)];
    owner[p] = c == Color::Black ? 1 : (c == Color::White ? -1 : 0);
  }
  bool visited[kMaxPoints] = {};
  int stack[kMaxPoints];
  int region[kMaxPoints];
  for (int p0 = 0; p0 < nn; ++p0) {
    if (visited[p0] || cells_[locOfPoint(p0, n_)] != Color::Empty) continue;
    int sp = 0, count = 0;
    bool touchB = false, touchW = false;
    stack[sp++] = p0;
    visited[p0] = true;
    while (sp > 0) {
      int p = stack[--sp];
      region[count++] = p;
      int loc = locOfPoint(p, n_);
      for (int d : kDirs) {
        int nb = loc + d;
        Color nc = cells_[nb];
        if (nc == Color::Black) touchB = true;
        else if (nc == Color::White) touchW = true;
        else if (nc == Color::Empty) {
          int q = pointOfLoc(nb, n_);
          if (!visited[q]) {
            visited[q] = true;
            stack[sp++] = q;
          }
        }
      }
    }
    int8_t o = 0;
    if (touchB && !touchW) o = 1;
    else if (touchW && !touchB) o = -1;
    for (int i = 0; i < count; ++i) owner[region[i]] = o;
  }
}

float Board::score() const {
  int8_t owner[kMaxPoints];
  areaOwnership(owner);
  int black = 0, white = 0;
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p) {
    if (owner[p] > 0) ++black;
    else if (owner[p] < 0) ++white;
  }
  return static_cast<float>(black - white) - komi_;
}

int Board::liberties(int p) const {
  int loc = locOfPoint(p, n_);
  if (chainId_[loc] < 0) return 0;
  return libs_[chainId_[loc]].count();
}

int Board::chainSize(int p) const {
  int loc = locOfPoint(p, n_);
  if (chainId_[loc] < 0) return 0;
  return chainSize_[chainId_[loc]];
}

// ---------------------------------------------------------------------------
// Symmetry, printing, comparison

Board Board::applySymmetry(int sym) const {
  Board b(n_, komi_, moveCap_);
  const int nn = n_ * n_;
  for (int p = 0; p < nn; ++p) b.cells_[locOfPoint(transformPoint(p, n_, sym), n_)] = cells_[locOfPoint(p, n_)];
  b.rebuildChains();
  b.toMove_ = toMove_;
  b.moveCount_ = moveCount_;
  b.passes_ = passes_;
  b.lastMove_ = transformMove(lastMove_, n_, sym);
  b.snapHead_ = snapHead_;
  b.snapCount_ = snapCount_;
  for (int i = 0; i < kHistoryLen; ++i) {
    Snapshot& dst = b.snaps_[i];
    const Snapshot& src = snaps_[i];
    dst = Snapshot{};
    for (int col = 0; col < 2; ++col)
      for (int p = 0; p < nn; ++p)
        if (src.stones[col].test(p)) dst.stones[col].set(transformPoint(p, n_, sym));
  }
  return b;
}

std::string Board::toString() const {
  std::ostringstream os;
  for (int r = 0; r < n_; ++r) {
    for (int c = 0; c < n_; ++c) {
      os << colorChar(cells_[locOf(r, c)]);
      if (c + 1 < n_) os << ' ';
    }
    os << '\n';
  }
  return os.str();
}

bool Board::operator==(const Board& o) const {
  if (n_ != o.n_ || toMove_ != o.toMove_ || moveCount_ != o.moveCount_ || passes_ != o.passes_ || hash_ != o.hash_ ||
      snapCount_ != o.snapCount_ || komi_ != o.komi_ || moveCap_ != o.moveCap_)
    return false;
  for (int i = 0; i < kMaxCells; ++i)
    if (cells_[i] != o.cells_[i]) return false;
  for (int k = 0; k < snapCount_; ++k)
    if (!(snapshot(k) == o.snapshot(k))) return false;
  return true;
}

}  // namespace mango
