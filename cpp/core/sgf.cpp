#include "core/sgf.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace mango {

namespace {

std::string sgfCoord(Move m, int n) {
  if (m == kPass) return "";
  int r = rowOfPoint(m, n), c = colOfPoint(m, n);
  std::string s;
  s.push_back(static_cast<char>('a' + c));
  s.push_back(static_cast<char>('a' + r));
  return s;
}

std::string escapeText(const std::string& s) {
  std::string out;
  for (char ch : s) {
    if (ch == ']' || ch == '\\') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

// Recursive-descent SGF reader that follows the main line only: in every game tree
// the first child variation is taken and the remaining siblings are skipped (they are
// still parsed for well-formedness, but their properties are ignored).
class Parser {
 public:
  explicit Parser(const std::string& text) : t_(text) {}

  bool parse(SgfGame* out) {
    if (!parseGameTree(/*mainLine=*/true)) return false;
    if (!sawNode_) return false;
    *out = g_;
    return true;
  }

 private:
  void ws() {
    while (i_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[i_]))) ++i_;
  }

  bool parseGameTree(bool mainLine) {
    ws();
    if (i_ >= t_.size() || t_[i_] != '(') return false;
    ++i_;
    if (!parseSequence(mainLine)) return false;
    bool first = true;
    for (;;) {
      ws();
      if (i_ >= t_.size()) return false;
      if (t_[i_] == ')') {
        ++i_;
        return true;
      }
      if (t_[i_] != '(') return false;
      if (!parseGameTree(mainLine && first)) return false;
      first = false;
    }
  }

  bool parseSequence(bool mainLine) {
    ws();
    if (i_ >= t_.size() || t_[i_] != ';') return false;
    while (i_ < t_.size() && t_[i_] == ';') {
      ++i_;
      if (mainLine) sawNode_ = true;
      if (!parseNode(mainLine)) return false;
      ws();
    }
    return true;
  }

  bool parseNode(bool mainLine) {
    for (;;) {
      ws();
      if (i_ >= t_.size()) return false;
      const char ch = t_[i_];
      if (ch == ';' || ch == '(' || ch == ')') return true;
      if (!std::isalpha(static_cast<unsigned char>(ch))) return false;
      std::string ident;
      while (i_ < t_.size() && std::isalpha(static_cast<unsigned char>(t_[i_]))) ident.push_back(t_[i_++]);
      ws();
      if (i_ >= t_.size() || t_[i_] != '[') return false;
      std::vector<std::string> values;
      while (i_ < t_.size() && t_[i_] == '[') {
        ++i_;
        std::string v;
        while (i_ < t_.size() && t_[i_] != ']') {
          if (t_[i_] == '\\' && i_ + 1 < t_.size()) ++i_;
          v.push_back(t_[i_++]);
        }
        if (i_ >= t_.size()) return false;
        ++i_;
        values.push_back(v);
        ws();
      }
      if (mainLine && !apply(ident, values.front())) return false;
    }
  }

  bool apply(const std::string& ident, const std::string& v) {
    if (ident == "SZ") g_.size = std::atoi(v.c_str());
    else if (ident == "KM") g_.komi = static_cast<float>(std::atof(v.c_str()));
    else if (ident == "RE") g_.result = v;
    else if (ident == "PB") g_.blackName = v;
    else if (ident == "PW") g_.whiteName = v;
    else if (ident == "GC") g_.comment = v;
    else if (ident == "B" || ident == "W") {
      if (v.empty() || (v == "tt" && g_.size <= 19)) {
        g_.moves.push_back(kPass);
      } else {
        if (v.size() != 2) return false;
        int c = v[0] - 'a', r = v[1] - 'a';
        if (c < 0 || r < 0 || c >= g_.size || r >= g_.size) return false;
        g_.moves.push_back(static_cast<Move>(pointOf(r, c, g_.size)));
      }
    }
    return true;
  }

  const std::string& t_;
  size_t i_ = 0;
  SgfGame g_;
  bool sawNode_ = false;
};

}  // namespace

std::string writeSgf(const SgfGame& g) {
  std::ostringstream os;
  os << "(;GM[1]FF[4]CA[UTF-8]AP[mango]RU[Tromp-Taylor]SZ[" << g.size << "]KM[" << g.komi << "]";
  if (!g.blackName.empty()) os << "PB[" << escapeText(g.blackName) << "]";
  if (!g.whiteName.empty()) os << "PW[" << escapeText(g.whiteName) << "]";
  if (!g.result.empty()) os << "RE[" << g.result << "]";
  if (!g.comment.empty()) os << "GC[" << escapeText(g.comment) << "]";
  Color c = Color::Black;
  for (Move m : g.moves) {
    os << ";" << (c == Color::Black ? 'B' : 'W') << "[" << sgfCoord(m, g.size) << "]";
    c = opposite(c);
  }
  os << ")\n";
  return os.str();
}

bool parseSgf(const std::string& text, SgfGame* out) {
  Parser p(text);
  return p.parse(out);
}

std::string resultString(float blackMinusWhiteScore, bool resigned, Color winnerIfResigned) {
  if (resigned) return winnerIfResigned == Color::Black ? "B+R" : "W+R";
  if (blackMinusWhiteScore == 0.0f) return "0";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%c+%g", blackMinusWhiteScore > 0 ? 'B' : 'W',
                blackMinusWhiteScore > 0 ? blackMinusWhiteScore : -blackMinusWhiteScore);
  return buf;
}

}  // namespace mango
