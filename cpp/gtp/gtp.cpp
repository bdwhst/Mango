#include "gtp/gtp.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>
#include <sstream>

#include "core/sgf.h"

namespace mango {

namespace {

const char* kLetters = "ABCDEFGHJKLMNOPQRST";  // no I

std::string lower(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

bool parseColor(const std::string& s, Color* c) {
  std::string l = lower(s);
  if (l == "b" || l == "black") {
    *c = Color::Black;
    return true;
  }
  if (l == "w" || l == "white") {
    *c = Color::White;
    return true;
  }
  return false;
}

const char* kCommands[] = {"protocol_version", "name",         "version",     "known_command", "list_commands",
                           "quit",             "boardsize",    "clear_board", "komi",          "play",
                           "genmove",          "undo",         "final_score", "showboard",     "mango-analyze"};

}  // namespace

Move RandomPlayer::genmove(const Board& board, const GameHistory& hist) {
  if (board.gameOver()) return kPass;
  board.legalMoves(HashHistory(hist), legal_);
  if (legal_.size() == 1) return kPass;
  return legal_[rng_.uniformInt(static_cast<uint32_t>(legal_.size() - 1))];
}

GtpEngine::GtpEngine(std::unique_ptr<Player> player, int size, float komi, std::optional<float> modelKomi)
    : player_(std::move(player)), size_(size), komi_(komi), modelKomi_(modelKomi), board_(size, komi) {
  hist_.reset(board_.hash());
}

void GtpEngine::newGame(int size) {
  size_ = size;
  board_ = Board(size_, komi_);
  hist_.reset(board_.hash());
  colors_.clear();
  player_->reset();
}

bool GtpEngine::playMove(Color c, Move m) {
  const Color before = board_.toMove();
  board_.setToMove(c);
  if (!board_.isLegal(m, HashHistory(hist_))) {
    board_.setToMove(before);
    return false;
  }
  board_.play(m);
  hist_.push(m, board_.hash());
  colors_.push_back(c);
  return true;
}

void GtpEngine::replayAll() {
  // Rebuild from the recorded (colour, move) sequence; colours need not alternate.
  std::vector<Move> moves = hist_.moves;
  std::vector<Color> colors = colors_;
  board_ = Board(size_, komi_);
  hist_.reset(board_.hash());
  colors_.clear();
  for (size_t i = 0; i < moves.size(); ++i) {
    if (!playMove(colors[i], moves[i])) break;  // cannot happen: the sequence was legal when played
  }
  player_->reset();
}

std::string GtpEngine::vertexToString(Move m, int n) {
  if (m == kPass) return "pass";
  int r = rowOfPoint(m, n), c = colOfPoint(m, n);
  std::string s;
  s.push_back(kLetters[c]);
  s += std::to_string(n - r);
  return s;
}

bool GtpEngine::parseVertex(const std::string& s, int n, Move* out) {
  std::string l = lower(s);
  if (l == "pass") {
    *out = kPass;
    return true;
  }
  if (l.size() < 2) return false;
  char col = static_cast<char>(std::toupper(static_cast<unsigned char>(l[0])));
  const char* pos = std::strchr(kLetters, col);
  if (!pos || col == 'I') return false;
  int c = static_cast<int>(pos - kLetters);
  int row = std::atoi(l.c_str() + 1);
  if (row < 1 || row > n || c >= n) return false;
  *out = static_cast<Move>(pointOf(n - row, c, n));
  return true;
}

std::string GtpEngine::handle(const std::string& rawLine, bool* quit) {
  *quit = false;
  std::string line = rawLine;
  size_t hash = line.find('#');
  if (hash != std::string::npos) line.erase(hash);
  std::replace(line.begin(), line.end(), '\t', ' ');
  std::istringstream is(line);
  std::vector<std::string> tok;
  std::string t;
  while (is >> t) tok.push_back(t);
  if (tok.empty()) return "";
  std::string id;
  if (std::all_of(tok[0].begin(), tok[0].end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); })) {
    id = tok[0];
    tok.erase(tok.begin());
    if (tok.empty()) return "?" + (id.empty() ? "" : " " + id) + " empty command";
  }
  const std::string cmd = lower(tok[0]);
  auto ok = [&](const std::string& body) { return "=" + (id.empty() ? "" : " " + id) + (body.empty() ? "" : " " + body); };
  auto err = [&](const std::string& body) { return "?" + (id.empty() ? "" : " " + id) + " " + body; };

  if (cmd == "protocol_version") return ok("2");
  if (cmd == "name") return ok(player_->name());
  if (cmd == "version") return ok("0.1");
  if (cmd == "quit") {
    *quit = true;
    return ok("");
  }
  if (cmd == "list_commands") {
    std::string s;
    for (const char* c : kCommands) {
      if (!s.empty()) s += "\n";
      s += c;
    }
    return ok(s);
  }
  if (cmd == "known_command") {
    if (tok.size() < 2) return err("syntax error");
    for (const char* c : kCommands)
      if (lower(tok[1]) == c) return ok("true");
    return ok("false");
  }
  if (cmd == "boardsize") {
    if (tok.size() < 2) return err("syntax error");
    int n = std::atoi(tok[1].c_str());
    if (n < kMinN || n > kMaxN) return err("unacceptable size");
    if (!player_->acceptsBoardSize(n)) return err("unacceptable size: this model plays " + std::to_string(size_) + "x" + std::to_string(size_));
    newGame(n);
    return ok("");
  }
  if (cmd == "clear_board") {
    newGame(size_);
    return ok("");
  }
  if (cmd == "komi") {
    if (tok.size() < 2) return err("syntax error");
    float k = static_cast<float>(std::atof(tok[1].c_str()));
    if (modelKomi_ && k != *modelKomi_)
      return err("unsupported komi " + tok[1] + "; this model was trained with " + std::to_string(*modelKomi_));
    komi_ = k;
    replayAll();
    return ok("");
  }
  if (cmd == "play") {
    if (tok.size() < 3) return err("syntax error");
    Color c;
    Move m;
    if (!parseColor(tok[1], &c)) return err("invalid color");
    if (!parseVertex(tok[2], size_, &m)) return err("invalid vertex");
    if (!playMove(c, m)) return err("illegal move");
    return ok("");
  }
  if (cmd == "genmove") {
    if (tok.size() < 2) return err("syntax error");
    Color c;
    if (!parseColor(tok[1], &c)) return err("invalid color");
    if (!player_->acceptsBoardSize(size_)) return err("board size " + std::to_string(size_) + " not supported by this player");
    if (board_.toMove() != c) {
      board_.setToMove(c);
      player_->reset();
    }
    Move m = player_->genmove(board_, hist_);
    if (!playMove(c, m)) {
      playMove(c, kPass);
      m = kPass;
    }
    return ok(vertexToString(m, size_));
  }
  if (cmd == "undo") {
    if (hist_.moves.empty()) return err("cannot undo");
    hist_.moves.pop_back();
    colors_.pop_back();
    replayAll();
    return ok("");
  }
  if (cmd == "final_score") return ok(resultString(board_.score(), false, Color::Black));
  if (cmd == "showboard") {
    std::ostringstream os;
    os << "\n";
    for (int r = 0; r < size_; ++r) {
      os << (size_ - r < 10 ? " " : "") << (size_ - r) << " ";
      for (int c = 0; c < size_; ++c) os << colorChar(board_.get(r, c)) << ' ';
      os << "\n";
    }
    os << "   ";
    for (int c = 0; c < size_; ++c) os << kLetters[c] << ' ';
    return ok(os.str());
  }
  if (cmd == "mango-analyze") {
    if (!player_->acceptsBoardSize(size_)) return err("board size " + std::to_string(size_) + " not supported by this player");
    return ok(player_->analyze(board_, hist_));
  }
  return err("unknown command");
}

int GtpEngine::run(std::istream& in, std::ostream& out) {
  std::string line;
  bool quit = false;
  while (!quit && std::getline(in, line)) {
    std::string resp = handle(line, &quit);
    if (resp.empty()) continue;
    out << resp << "\n\n" << std::flush;
  }
  return 0;
}

}  // namespace mango
