#include "nn/model_meta.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/types.h"
#include "nlohmann/json.hpp"

namespace mango {

using nlohmann::json;

ModelMeta ModelMeta::fromJsonText(const std::string& text) {
  json j = json::parse(text);
  ModelMeta m;
  auto get = [&](const char* key, auto& out) {
    if (!j.contains(key)) throw std::runtime_error(std::string("model.json missing field: ") + key);
    out = j.at(key).get<std::remove_reference_t<decltype(out)>>();
  };
  get("format", m.format);
  get("model_id", m.modelId);
  get("board_size", m.boardSize);
  get("planes", m.planes);
  get("feature_schema", m.featureSchema);
  get("rules_id", m.rulesId);
  get("komi", m.komi);
  get("move_cap", m.moveCap);
  get("res_blocks", m.resBlocks);
  get("filters", m.filters);
  if (j.contains("export_dtype")) m.exportDtype = j["export_dtype"].get<std::string>();
  if (j.contains("config_fingerprint")) m.configFingerprint = j["config_fingerprint"].get<std::string>();
  if (m.format != "torchscript-v1") throw std::runtime_error("unsupported model format: " + m.format);
  return m;
}

ModelMeta ModelMeta::load(const std::string& modelDir) {
  std::ifstream in(modelDir + "/model.json");
  if (!in) throw std::runtime_error("cannot open " + modelDir + "/model.json");
  std::stringstream ss;
  ss << in.rdbuf();
  return fromJsonText(ss.str());
}

void ModelMeta::validate(const BoardConfig& board, bool allowKomiMismatch) const {
  if (boardSize != board.size)
    throw std::runtime_error("model board_size " + std::to_string(boardSize) + " != config " + std::to_string(board.size));
  if (planes != kNumPlanes) throw std::runtime_error("model planes " + std::to_string(planes) + " != " + std::to_string(kNumPlanes));
  if (featureSchema != kFeatureSchema)
    throw std::runtime_error("model feature_schema " + std::to_string(featureSchema) + " != engine " + std::to_string(kFeatureSchema));
  if (rulesId != kRulesId) throw std::runtime_error("model rules_id " + std::to_string(rulesId) + " != engine " + std::to_string(kRulesId));
  if (komi != board.komi && !allowKomiMismatch)
    throw std::runtime_error("model komi " + std::to_string(komi) + " != config komi " + std::to_string(board.komi) +
                             " (pass --allow-komi-mismatch to override)");
}

}  // namespace mango
