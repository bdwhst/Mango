#include "selfplay/game_runner.h"

#include <vector>

#include "core/random.h"

namespace mango {

namespace {

// Counts positions sent to the wrapped evaluator.
class CountingEvaluator : public NNEvaluator {
 public:
  explicit CountingEvaluator(NNEvaluator& inner) : inner_(inner) {}
  int boardSize() const override { return inner_.boardSize(); }
  const std::string& modelId() const override { return inner_.modelId(); }
  void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) override {
    positions_ += static_cast<int>(in.size());
    inner_.evaluate(in, out);
  }
  int positions() const { return positions_; }

 private:
  NNEvaluator& inner_;
  int positions_ = 0;
};

}  // namespace

bool isNoResignGame(uint64_t gameSeed, float fraction) {
  if (fraction <= 0.0f) return false;
  // Independent of the holdout tag (game_seed % 20), which uses the raw seed.
  const uint64_t r = deriveSeed(gameSeed, 0x6E6F726573696E67ull) % 10000;
  return static_cast<double>(r) < static_cast<double>(fraction) * 10000.0;
}

SelfplayGame::SelfplayGame(const SelfplayGameOptions& opt, std::string modelId)
    : opt_(opt), modelId_(std::move(modelId)), board_(opt.boardSize, opt.komi, opt.moveCap) {
  hist_.reset(board_.hash());
  tree_ = std::make_unique<SearchTree>(opt_.params, opt_.boardSize, opt_.gameSeed);
  tree_->newGame(board_, hist_);
  record_.gameSeed = opt_.gameSeed;
  record_.noResignGame = opt_.noResignGame;
  snapshot();
  if (board_.gameOver()) finished_ = true;  // move cap 0: nothing to play
}

void SelfplayGame::snapshot() {
  const int n = board_.size();
  std::vector<Color> cells(n * n);
  for (int p = 0; p < n * n; ++p) cells[p] = board_.atPoint(p);
  record_.snapshots.emplace_back(packedSnapshotBytes(n));
  packSnapshot(cells.data(), n, record_.snapshots.back().data());
}

bool SelfplayGame::finishMove() {
  if (finished_) return true;
  const int nn = board_.numPoints();
  const float v = tree_->rootValue();
  const float maxQ = tree_->rootMaxQ();
  const bool resignEnabled = opt_.params.resignThreshold > -1.0f && !opt_.noResignGame;
  if (resignEnabled && tree_->shouldResign()) {
    resigned_ = true;
    resigner_ = board_.toMove();
    finished_ = true;
    return true;
  }
  const Move m = tree_->selectMove(board_.moveCount());
  MoveVisits mv;
  mv.rootTotalVisits = tree_->rootTotalVisits();
  std::vector<std::pair<Move, uint32_t>> rv;
  tree_->rootVisits(rv);
  for (const auto& [move, cnt] : rv) mv.counts.emplace_back(static_cast<uint16_t>(move == kPass ? nn : move), cnt);
  record_.moves.push_back(static_cast<uint16_t>(m == kPass ? nn : m));
  record_.rootValue.push_back(v);
  record_.rootMaxQ.push_back(maxQ);
  record_.visits.push_back(std::move(mv));
  if (opt_.storeSearchKind) record_.searchKind.push_back(1);
  board_.play(m);
  hist_.push(m, board_.hash());
  tree_->advance(m, board_, hist_);
  snapshot();
  if (board_.gameOver()) finished_ = true;
  return finished_;
}

SelfplayGameResult SelfplayGame::takeResult(int evaluations) {
  const int n = board_.size();
  const int nn = n * n;
  SelfplayGameResult res;
  GameRecord& g = record_;
  g.score = board_.score();
  if (resigned_) {
    g.termination = Termination::Resign;
    g.result = resigner_ == Color::Black ? -1 : 1;
  } else {
    g.termination = board_.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
    g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
  }
  if (opt_.storeFinalOwnership) {
    g.finalOwnership.resize(nn);
    board_.areaOwnership(g.finalOwnership.data());
  }
  res.sgf.size = n;
  res.sgf.komi = opt_.komi;
  for (uint16_t m : g.moves) res.sgf.moves.push_back(m == nn ? kPass : static_cast<Move>(m));
  res.sgf.result = resultString(g.score, resigned_, resigned_ ? opposite(resigner_) : Color::Empty);
  res.sgf.blackName = "mango " + modelId_;
  res.sgf.whiteName = "mango " + modelId_;
  res.sgf.comment = "game_seed=" + std::to_string(opt_.gameSeed) + " model=" + modelId_ +
                    (opt_.noResignGame ? " no_resign=1" : " no_resign=0");
  res.evaluations = evaluations;
  res.record = std::move(record_);
  record_ = GameRecord();
  return res;
}

SelfplayGameResult playSelfplayGame(NNEvaluator& evIn, const SelfplayGameOptions& opt, const std::string& modelId) {
  CountingEvaluator ev(evIn);
  SelfplayGame game(opt, modelId);
  while (!game.finished()) {
    game.tree().runSequential(ev);
    game.finishMove();
  }
  return game.takeResult(ev.positions());
}

}  // namespace mango
