// Search parameters (docs/DESIGN.md sections 5.4 and 8), derived from SearchConfig.
#pragma once

#include "core/config.h"

namespace mango {

struct SearchParams {
  enum class Fpu { Zero, Parent };

  int simulations = 200;            // new, completed simulations per move (the budget)
  float cPuct = 1.5f;
  Fpu fpu = Fpu::Zero;
  float dirichletAlpha = 0.134f;
  float dirichletEpsilon = 0.25f;
  bool addRootNoise = false;        // self-play only
  int temperatureMoves = 8;         // tau = 1 below this move number, tau -> 0 after
  bool searchSymmetry = true;       // random dihedral symmetry per evaluation
  bool treeReuse = true;
  bool budgetIncludesInherited = false;
  float resignThreshold = -1.0f;    // r < threshold resigns; -1 disables (r >= -1 always)

  static SearchParams fromConfig(const SearchConfig& c, int boardSize, bool selfplay) {
    SearchParams p;
    p.simulations = selfplay ? c.simulations : c.evalSimulations;
    p.cPuct = c.cPuct;
    p.fpu = c.fpu == "parent" ? Fpu::Parent : Fpu::Zero;
    p.dirichletAlpha = c.effectiveDirichletAlpha(boardSize);
    p.dirichletEpsilon = c.dirichletEpsilon;
    p.addRootNoise = selfplay;
    p.temperatureMoves = selfplay ? c.temperatureMoves : 0;
    p.searchSymmetry = c.searchSymmetry;
    p.treeReuse = c.treeReuse;
    p.budgetIncludesInherited = c.budgetIncludesInherited;
    p.resignThreshold = selfplay ? c.resignThreshold : -1.0f;
    return p;
  }
};

}  // namespace mango
