#include "selfplay/game_runner.h"

#include <vector>

#include "core/random.h"
#include "search/mcts.h"

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

void snapshotOf(const Board& board, std::vector<uint8_t>& out) {
  const int n = board.size();
  std::vector<Color> cells(n * n);
  for (int p = 0; p < n * n; ++p) cells[p] = board.atPoint(p);
  out.resize(packedSnapshotBytes(n));
  packSnapshot(cells.data(), n, out.data());
}

}  // namespace

bool isNoResignGame(uint64_t gameSeed, float fraction) {
  if (fraction <= 0.0f) return false;
  // Independent of the holdout tag (game_seed % 20), which uses the raw seed.
  const uint64_t r = deriveSeed(gameSeed, 0x6E6F726573696E67ull) % 10000;
  return static_cast<double>(r) < static_cast<double>(fraction) * 10000.0;
}

SelfplayGameResult playSelfplayGame(NNEvaluator& evIn, const SelfplayGameOptions& opt, const std::string& modelId) {
  const int n = opt.boardSize;
  const int nn = n * n;
  CountingEvaluator ev(evIn);
  Board board(n, opt.komi, opt.moveCap);
  GameHistory hist;
  hist.reset(board.hash());
  SearchTree tree(opt.params, n, opt.gameSeed);
  tree.newGame(board, hist);

  SelfplayGameResult res;
  GameRecord& g = res.record;
  g.gameSeed = opt.gameSeed;
  g.noResignGame = opt.noResignGame;
  g.snapshots.emplace_back();
  snapshotOf(board, g.snapshots.back());

  bool resigned = false;
  Color resigner = Color::Empty;
  const bool resignEnabled = opt.params.resignThreshold > -1.0f && !opt.noResignGame;
  while (!board.gameOver()) {
    tree.runSequential(ev);
    const float v = tree.rootValue();
    const float maxQ = tree.rootMaxQ();
    if (resignEnabled && tree.shouldResign()) {
      resigned = true;
      resigner = board.toMove();
      break;
    }
    const Move m = tree.selectMove(board.moveCount());
    MoveVisits mv;
    mv.rootTotalVisits = tree.rootTotalVisits();
    std::vector<std::pair<Move, uint32_t>> rv;
    tree.rootVisits(rv);
    for (const auto& [mv_move, cnt] : rv) mv.counts.emplace_back(static_cast<uint16_t>(mv_move == kPass ? nn : mv_move), cnt);
    g.moves.push_back(static_cast<uint16_t>(m == kPass ? nn : m));
    g.rootValue.push_back(v);
    g.rootMaxQ.push_back(maxQ);
    g.visits.push_back(std::move(mv));
    if (opt.storeSearchKind) g.searchKind.push_back(1);
    board.play(m);
    hist.push(m, board.hash());
    tree.advance(m, board, hist);
    g.snapshots.emplace_back();
    snapshotOf(board, g.snapshots.back());
  }

  g.score = board.score();
  if (resigned) {
    g.termination = Termination::Resign;
    g.result = resigner == Color::Black ? -1 : 1;
  } else {
    g.termination = board.consecutivePasses() >= 2 ? Termination::TwoPasses : Termination::MoveCap;
    g.result = g.score > 0 ? 1 : (g.score < 0 ? -1 : 0);
  }
  if (opt.storeFinalOwnership) {
    g.finalOwnership.resize(nn);
    board.areaOwnership(g.finalOwnership.data());
  }

  res.sgf.size = n;
  res.sgf.komi = opt.komi;
  for (uint16_t m : g.moves) res.sgf.moves.push_back(m == nn ? kPass : static_cast<Move>(m));
  res.sgf.result = resultString(g.score, resigned, resigned ? opposite(resigner) : Color::Empty);
  res.sgf.blackName = "mango " + modelId;
  res.sgf.whiteName = "mango " + modelId;
  res.sgf.comment = "game_seed=" + std::to_string(opt.gameSeed) + " model=" + modelId +
                    (opt.noResignGame ? " no_resign=1" : " no_resign=0");
  res.evaluations = ev.positions();
  return res;
}

}  // namespace mango
