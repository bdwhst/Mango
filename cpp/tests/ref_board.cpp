#include "tests/ref_board.h"

#include <algorithm>

#include "core/zobrist.h"

namespace mango::test {

RefBoard::RefBoard(int n_, float komi_, int moveCap_)
    : n(n_), komi(komi_), moveCap(moveCap_ < 0 ? 2 * n_ * n_ : moveCap_), cells(n_ * n_, Color::Empty),
      toMove(Color::Black), passes(0), moveCount(0) {
  positions.push_back(cells);
}

void RefBoard::chainOf(const std::vector<Color>& cs, int p, std::vector<int>& chain, std::vector<int>& libs) const {
  chain.clear();
  libs.clear();
  const Color c = cs[p];
  std::vector<char> seen(n * n, 0);
  std::vector<int> stack{p};
  seen[p] = 1;
  while (!stack.empty()) {
    int q = stack.back();
    stack.pop_back();
    chain.push_back(q);
    int r = q / n, col = q % n;
    const int dr[4] = {0, 0, 1, -1}, dc[4] = {1, -1, 0, 0};
    for (int d = 0; d < 4; ++d) {
      int rr = r + dr[d], cc = col + dc[d];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
      int nb = rr * n + cc;
      if (cs[nb] == c && !seen[nb]) {
        seen[nb] = 1;
        stack.push_back(nb);
      } else if (cs[nb] == Color::Empty && std::find(libs.begin(), libs.end(), nb) == libs.end()) {
        libs.push_back(nb);
      }
    }
  }
}

bool RefBoard::simulate(Move m, std::vector<Color>& out) const {
  out = cells;
  if (m == kPass) return true;
  if (out[m] != Color::Empty) return false;
  const Color c = toMove;
  const Color enemy = opposite(c);
  out[m] = c;
  // Remove enemy chains without liberties.
  std::vector<int> chain, libs;
  int r = m / n, col = m % n;
  const int dr[4] = {0, 0, 1, -1}, dc[4] = {1, -1, 0, 0};
  for (int d = 0; d < 4; ++d) {
    int rr = r + dr[d], cc = col + dc[d];
    if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
    int nb = rr * n + cc;
    if (out[nb] != enemy) continue;
    chainOf(out, nb, chain, libs);
    if (libs.empty())
      for (int q : chain) out[q] = Color::Empty;
  }
  chainOf(out, m, chain, libs);
  if (libs.empty()) return false;  // suicide
  return true;
}

bool RefBoard::isLegal(Move m) const {
  if (m == kPass) return true;
  if (m < 0 || m >= n * n) return false;
  std::vector<Color> out;
  if (!simulate(m, out)) return false;
  for (const auto& pos : positions)
    if (pos == out) return false;
  return true;
}

void RefBoard::play(Move m) {
  std::vector<Color> out;
  bool ok = simulate(m, out);
  (void)ok;
  cells = out;
  if (m == kPass) ++passes;
  else passes = 0;
  ++moveCount;
  toMove = opposite(toMove);
  positions.push_back(cells);
}

int RefBoard::liberties(int p) const {
  if (cells[p] == Color::Empty) return 0;
  std::vector<int> chain, libs;
  chainOf(cells, p, chain, libs);
  return static_cast<int>(libs.size());
}

int RefBoard::chainSize(int p) const {
  if (cells[p] == Color::Empty) return 0;
  std::vector<int> chain, libs;
  chainOf(cells, p, chain, libs);
  return static_cast<int>(chain.size());
}

float RefBoard::score() const {
  int black = 0, white = 0;
  std::vector<char> seen(n * n, 0);
  const int dr[4] = {0, 0, 1, -1}, dc[4] = {1, -1, 0, 0};
  for (int p = 0; p < n * n; ++p) {
    if (cells[p] == Color::Black) ++black;
    else if (cells[p] == Color::White) ++white;
    else if (!seen[p]) {
      std::vector<int> region, stack{p};
      seen[p] = 1;
      bool tb = false, tw = false;
      while (!stack.empty()) {
        int q = stack.back();
        stack.pop_back();
        region.push_back(q);
        int r = q / n, col = q % n;
        for (int d = 0; d < 4; ++d) {
          int rr = r + dr[d], cc = col + dc[d];
          if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
          int nb = rr * n + cc;
          if (cells[nb] == Color::Black) tb = true;
          else if (cells[nb] == Color::White) tw = true;
          else if (!seen[nb]) {
            seen[nb] = 1;
            stack.push_back(nb);
          }
        }
      }
      if (tb && !tw) black += static_cast<int>(region.size());
      else if (tw && !tb) white += static_cast<int>(region.size());
    }
  }
  return static_cast<float>(black - white) - komi;
}

std::vector<uint8_t> RefBoard::features() const {
  const int nn = n * n;
  std::vector<uint8_t> f(kNumPlanes * nn, 0);
  const Color me = toMove, opp = opposite(toMove);
  for (int k = 0; k < kHistoryLen; ++k) {
    int idx = static_cast<int>(positions.size()) - 1 - k;
    if (idx < 0) break;
    const auto& pos = positions[idx];
    for (int p = 0; p < nn; ++p) {
      if (pos[p] == me) f[k * nn + p] = 1;
      else if (pos[p] == opp) f[(kHistoryLen + k) * nn + p] = 1;
    }
  }
  if (toMove == Color::Black)
    for (int p = 0; p < nn; ++p) f[2 * kHistoryLen * nn + p] = 1;
  return f;
}

uint64_t RefBoard::hash() const {
  uint64_t h = 0;
  for (int p = 0; p < n * n; ++p)
    if (cells[p] == Color::Black || cells[p] == Color::White) h ^= zobristKey(cells[p], locOfPoint(p, n));
  return h;
}

}  // namespace mango::test
