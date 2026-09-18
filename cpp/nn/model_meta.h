// model.json of a model version directory (docs/DESIGN.md section 6.4) and the
// consistency checks between a model and a run configuration.
#pragma once

#include <string>

#include "core/config.h"

namespace mango {

struct ModelMeta {
  std::string format;
  std::string modelId;
  int boardSize = 0;
  int planes = 17;
  int featureSchema = 1;
  int rulesId = 1;
  float komi = 7.5f;
  int moveCap = 0;
  int resBlocks = 0;
  int filters = 0;
  std::string exportDtype;
  std::string configFingerprint;

  static ModelMeta load(const std::string& modelDir);  // reads <modelDir>/model.json
  static ModelMeta fromJsonText(const std::string& text);

  // Throws std::runtime_error with a precise message if the model cannot be used with
  // this board/rules configuration. Komi mismatch throws unless allowKomiMismatch.
  void validate(const BoardConfig& board, bool allowKomiMismatch = false) const;
};

}  // namespace mango
