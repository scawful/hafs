#include "data_loader.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

// Use nlohmann/json for JSON parsing (simpler than simdjson for this use case)
// If simdjson is available, we could optimize later
#include <filesystem>

namespace fs = std::filesystem;

namespace hafs {
namespace viz {

namespace {

constexpr size_t kTrendWindow = 5;
constexpr float kTrendDeltaThreshold = 0.05f;
constexpr float kSparseCoverageFactor = 0.5f;

// Simple JSON value types for basic parsing
// This is a minimal JSON parser for when nlohmann/json isn't available
// In production, we'd use a proper JSON library

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;

  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  JsonArray array_value;
  JsonObject object_value;

  bool IsNull() const { return type == Type::Null; }
  bool IsBool() const { return type == Type::Bool; }
  bool IsNumber() const { return type == Type::Number; }
  bool IsString() const { return type == Type::String; }
  bool IsArray() const { return type == Type::Array; }
  bool IsObject() const { return type == Type::Object; }

  int GetInt(int default_val = 0) const {
    return IsNumber() ? static_cast<int>(number_value) : default_val;
  }
  float GetFloat(float default_val = 0.0f) const {
    return IsNumber() ? static_cast<float>(number_value) : default_val;
  }
  std::string GetString(const std::string& default_val = "") const {
    return IsString() ? string_value : default_val;
  }

  const JsonValue& operator[](const std::string& key) const {
    static JsonValue null_value;
    if (!IsObject()) return null_value;
    auto it = object_value.find(key);
    return it != object_value.end() ? it->second : null_value;
  }

  const JsonValue& operator[](size_t index) const {
    static JsonValue null_value;
    if (!IsArray() || index >= array_value.size()) return null_value;
    return array_value[index];
  }
};

struct ParseContext {
  bool ok = true;
  std::string error;
};

void SetParseError(ParseContext* ctx, const std::string& error) {
  if (!ctx || !ctx->ok) return;
  ctx->ok = false;
  ctx->error = error;
}

// Forward declarations for recursive parsing
JsonValue ParseValue(const std::string& json, size_t& pos, ParseContext* ctx);

void SkipWhitespace(const std::string& json, size_t& pos) {
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
}

std::string ParseString(const std::string& json, size_t& pos, ParseContext* ctx) {
  if (pos >= json.size() || json[pos] != '"') {
    SetParseError(ctx, "Expected string");
    return "";
  }
  ++pos;  // skip opening quote

  std::string result;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      ++pos;
      switch (json[pos]) {
        case 'n':
          result += '\n';
          break;
        case 't':
          result += '\t';
          break;
        case 'r':
          result += '\r';
          break;
        case '"':
          result += '"';
          break;
        case '\\':
          result += '\\';
          break;
        default:
          result += json[pos];
          break;
      }
    } else {
      result += json[pos];
    }
    ++pos;
  }

  if (pos < json.size()) {
    ++pos;  // skip closing quote
  } else {
    SetParseError(ctx, "Unterminated string");
  }
  return result;
}

double ParseNumber(const std::string& json, size_t& pos, ParseContext* ctx) {
  size_t start = pos;
  if (json[pos] == '-') ++pos;
  bool has_digit = false;
  while (pos < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[pos])) ||
          json[pos] == '.' ||
                               json[pos] == 'e' || json[pos] == 'E' ||
                               json[pos] == '+' || json[pos] == '-')) {
    if (std::isdigit(static_cast<unsigned char>(json[pos]))) has_digit = true;
    if ((json[pos] == 'e' || json[pos] == 'E') && pos > start) {
      ++pos;
      if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) ++pos;
    } else {
      ++pos;
    }
  }
  if (!has_digit) {
    SetParseError(ctx, "Invalid number");
    return 0.0;
  }
  try {
    return std::stod(json.substr(start, pos - start));
  } catch (const std::exception&) {
    SetParseError(ctx, "Invalid number");
    return 0.0;
  }
}

JsonValue ParseArray(const std::string& json, size_t& pos, ParseContext* ctx) {
  JsonValue result;
  result.type = JsonValue::Type::Array;

  ++pos;  // skip '['
  SkipWhitespace(json, pos);

  while (pos < json.size() && json[pos] != ']') {
    result.array_value.push_back(ParseValue(json, pos, ctx));
    if (ctx && !ctx->ok) break;
    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == ',') {
      ++pos;
      SkipWhitespace(json, pos);
    }
  }

  if (pos < json.size()) {
    ++pos;  // skip ']'
  } else {
    SetParseError(ctx, "Unterminated array");
  }
  return result;
}

JsonValue ParseObject(const std::string& json, size_t& pos, ParseContext* ctx) {
  JsonValue result;
  result.type = JsonValue::Type::Object;

  ++pos;  // skip '{'
  SkipWhitespace(json, pos);

  while (pos < json.size() && json[pos] != '}') {
    std::string key = ParseString(json, pos, ctx);
    if (ctx && !ctx->ok) break;
    SkipWhitespace(json, pos);

    if (pos < json.size() && json[pos] == ':') {
      ++pos;
      SkipWhitespace(json, pos);
    }

    result.object_value[key] = ParseValue(json, pos, ctx);
    if (ctx && !ctx->ok) break;
    SkipWhitespace(json, pos);

    if (pos < json.size() && json[pos] == ',') {
      ++pos;
      SkipWhitespace(json, pos);
    }
  }

  if (pos < json.size()) {
    ++pos;  // skip '}'
  } else {
    SetParseError(ctx, "Unterminated object");
  }
  return result;
}

JsonValue ParseValue(const std::string& json, size_t& pos, ParseContext* ctx) {
  SkipWhitespace(json, pos);

  if (pos >= json.size()) return JsonValue{};

  char c = json[pos];

  if (c == '{') {
    return ParseObject(json, pos, ctx);
  } else if (c == '[') {
    return ParseArray(json, pos, ctx);
  } else if (c == '"') {
    JsonValue v;
    v.type = JsonValue::Type::String;
    v.string_value = ParseString(json, pos, ctx);
    return v;
  } else if (c == 't' && json.substr(pos, 4) == "true") {
    pos += 4;
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    v.bool_value = true;
    return v;
  } else if (c == 'f' && json.substr(pos, 5) == "false") {
    pos += 5;
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    v.bool_value = false;
    return v;
  } else if (c == 'n' && json.substr(pos, 4) == "null") {
    pos += 4;
    return JsonValue{};
  } else if (c == '-' || std::isdigit(c)) {
    JsonValue v;
    v.type = JsonValue::Type::Number;
    v.number_value = ParseNumber(json, pos, ctx);
    return v;
  }

  SetParseError(ctx, "Unexpected token");
  ++pos;
  return JsonValue{};
}

JsonValue ParseJson(const std::string& json, std::string* error) {
  size_t pos = 0;
  ParseContext ctx;
  JsonValue value = ParseValue(json, pos, &ctx);
  SkipWhitespace(json, pos);
  if (ctx.ok && pos < json.size()) {
    SetParseError(&ctx, "Trailing data after JSON document");
  }
  if (error) *error = ctx.ok ? "" : ctx.error;
  return value;
}

bool ReadFile(const std::string& path, std::string* content, std::string* error) {
  std::ifstream file(path);
  if (!file.is_open()) {
    if (error) *error = "Failed to open file";
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  if (!buffer) {
    if (error) *error = "Failed to read file";
    return false;
  }
  if (content) *content = buffer.str();
  return true;
}

bool PathExistsDefault(const std::string& path) {
  return fs::exists(path);
}

bool IsWhitespaceOnly(const std::string& text) {
  return std::all_of(text.begin(), text.end(),
                     [](unsigned char ch) { return std::isspace(ch); });
}

}  // namespace

DataLoader::DataLoader(const std::string& data_path,
                       FileReader file_reader,
                       PathExists path_exists)
    : data_path_(data_path),
      file_reader_(file_reader ? std::move(file_reader) : ReadFile),
      path_exists_(path_exists ? std::move(path_exists) : PathExistsDefault) {}

bool DataLoader::Refresh() {
  last_error_.clear();
  last_status_ = LoadStatus{};

  if (!path_exists_(data_path_)) {
    last_error_ = "Data path does not exist: " + data_path_;
    last_status_.error_count = 1;
    last_status_.last_error = last_error_;
    last_status_.last_error_source = "data_path";
    return false;
  }

  auto next_quality_trends = quality_trends_;
  auto next_generator_stats = generator_stats_;
  auto next_rejection_summary = rejection_summary_;
  auto next_embedding_regions = embedding_regions_;
  auto next_coverage = coverage_;
  auto next_training_runs = training_runs_;
  auto next_optimization_data = optimization_data_;

  LoadResult quality = LoadQualityFeedback(&next_quality_trends,
                                           &next_generator_stats,
                                           &next_rejection_summary);
  last_status_.quality_found = quality.found;
  last_status_.quality_ok = quality.ok;
  if (quality.found && !quality.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = quality.error;
      last_status_.last_error_source = "quality_feedback.json";
    }
  }
  if (quality.ok) {
    quality_trends_ = std::move(next_quality_trends);
    generator_stats_ = std::move(next_generator_stats);
    rejection_summary_ = std::move(next_rejection_summary);
  }

  LoadResult active = LoadActiveLearning(&next_embedding_regions, &next_coverage);
  last_status_.active_found = active.found;
  last_status_.active_ok = active.ok;
  if (active.found && !active.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = active.error;
      last_status_.last_error_source = "active_learning.json";
    }
  }
  if (active.ok) {
    embedding_regions_ = std::move(next_embedding_regions);
    coverage_ = std::move(next_coverage);
  }

  LoadResult training = LoadTrainingFeedback(&next_training_runs,
                                             &next_optimization_data);
  last_status_.training_found = training.found;
  last_status_.training_ok = training.ok;
  if (training.found && !training.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = training.error;
      last_status_.last_error_source = "training_feedback.json";
    }
  }
  if (training.ok) {
    training_runs_ = std::move(next_training_runs);
    optimization_data_ = std::move(next_optimization_data);
  }

  has_data_ = !quality_trends_.empty() || !generator_stats_.empty() ||
              !embedding_regions_.empty() || !training_runs_.empty() ||
              !optimization_data_.domain_effectiveness.empty() ||
              !optimization_data_.threshold_sensitivity.empty();
  last_error_ = last_status_.last_error;

  bool any_found = last_status_.FoundCount() > 0;
  return last_status_.AnyOk() || (!any_found && has_data_);
}

DataLoader::LoadResult DataLoader::LoadQualityFeedback(
    std::vector<QualityTrendData>* quality_trends,
    std::vector<GeneratorStatsData>* generator_stats,
    RejectionSummary* rejection_summary) {
  LoadResult result;
  std::string path = data_path_ + "/quality_feedback.json";
  if (!path_exists_(path)) return result;

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty()
                       ? "quality_feedback.json is empty"
                       : ("quality_feedback.json: " + read_error);
    return result;
  }

  std::string parse_error;
  JsonValue data = ParseJson(content, &parse_error);
  if (!parse_error.empty()) {
    result.ok = false;
    result.error = "quality_feedback.json: " + parse_error;
    return result;
  }
  if (!data.IsObject()) {
    result.ok = false;
    result.error = "quality_feedback.json: invalid JSON root object";
    return result;
  }

  std::vector<QualityTrendData> next_quality_trends;
  std::vector<GeneratorStatsData> next_generator_stats;
  RejectionSummary next_rejection_summary;

  // Parse generator stats
  const auto& gen_stats = data["generator_stats"];
  if (gen_stats.IsObject()) {
    for (const auto& [name, stats] : gen_stats.object_value) {
      GeneratorStatsData gs;
      gs.name = name;
      gs.samples_generated = stats["samples_generated"].GetInt();
      gs.samples_accepted = stats["samples_accepted"].GetInt();
      gs.samples_rejected = stats["samples_rejected"].GetInt();
      gs.avg_quality = stats["avg_quality_score"].GetFloat();

      int total = gs.samples_accepted + gs.samples_rejected;
      gs.acceptance_rate = total > 0 ? static_cast<float>(gs.samples_accepted) /
                                           static_cast<float>(total)
                                     : 0.0f;

      // Parse rejection reasons
      const auto& reasons = stats["rejection_reasons"];
      if (reasons.IsObject()) {
        for (const auto& [reason, count] : reasons.object_value) {
          int c = count.GetInt();
          gs.rejection_reasons[reason] = c;
          next_rejection_summary.reasons[reason] += c;
          next_rejection_summary.total_rejections += c;
        }
      }

      next_generator_stats.push_back(std::move(gs));
    }
  }

  // Parse rejection history for trends
  const auto& history = data["rejection_history"];
  if (history.IsArray()) {
    std::map<std::pair<std::string, std::string>, QualityTrendData> trends_map;

    for (const auto& entry : history.array_value) {
      std::string domain = entry["domain"].GetString("unknown");
      const auto& scores = entry["scores"];

      if (scores.IsObject()) {
        for (const auto& [metric, value] : scores.object_value) {
          auto key = std::make_pair(domain, metric);
          if (trends_map.find(key) == trends_map.end()) {
            trends_map[key] = QualityTrendData{domain, metric};
          }
          trends_map[key].values.push_back(value.GetFloat());
        }
      }
    }

    // Calculate trend stats
    for (auto& [key, trend] : trends_map) {
      if (!trend.values.empty()) {
        float sum = 0.0f;
        for (float v : trend.values) sum += v;
        trend.mean = sum / static_cast<float>(trend.values.size());

        // Determine trend direction
        if (trend.values.size() < kTrendWindow) {
          trend.trend_direction = "insufficient";
        } else {
          float recent = 0.0f, older = 0.0f;
          for (size_t i = trend.values.size() - kTrendWindow;
               i < trend.values.size(); ++i) {
            recent += trend.values[i];
          }
          for (size_t i = 0; i < kTrendWindow; ++i) {
            older += trend.values[i];
          }
          recent /= static_cast<float>(kTrendWindow);
          older /= static_cast<float>(kTrendWindow);
          float diff = recent - older;

          if (diff > kTrendDeltaThreshold)
            trend.trend_direction = "improving";
          else if (diff < -kTrendDeltaThreshold)
            trend.trend_direction = "declining";
          else
            trend.trend_direction = "stable";
        }
      }
      next_quality_trends.push_back(std::move(trend));
    }
  }

  if (quality_trends) *quality_trends = std::move(next_quality_trends);
  if (generator_stats) *generator_stats = std::move(next_generator_stats);
  if (rejection_summary) *rejection_summary = std::move(next_rejection_summary);

  result.ok = true;
  return result;
}

DataLoader::LoadResult DataLoader::LoadActiveLearning(
    std::vector<EmbeddingRegionData>* embedding_regions,
    CoverageData* coverage) {
  LoadResult result;
  std::string path = data_path_ + "/active_learning.json";
  if (!path_exists_(path)) return result;

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty()
                       ? "active_learning.json is empty"
                       : ("active_learning.json: " + read_error);
    return result;
  }

  std::string parse_error;
  JsonValue data = ParseJson(content, &parse_error);
  if (!parse_error.empty()) {
    result.ok = false;
    result.error = "active_learning.json: " + parse_error;
    return result;
  }
  if (!data.IsObject()) {
    result.ok = false;
    result.error = "active_learning.json: invalid JSON root object";
    return result;
  }

  std::vector<EmbeddingRegionData> next_embedding_regions;
  CoverageData next_coverage;

  // Parse regions
  const auto& regions = data["regions"];
  if (regions.IsArray()) {
    int idx = 0;
    for (const auto& region : regions.array_value) {
      EmbeddingRegionData erd;
      erd.index = idx++;
      erd.sample_count = region["sample_count"].GetInt();
      erd.domain = region["domain"].GetString("unknown");
      erd.avg_quality = region["avg_quality"].GetFloat();
      next_embedding_regions.push_back(std::move(erd));
    }
  }

  // Calculate coverage stats
  if (!next_embedding_regions.empty()) {
    next_coverage.num_regions = static_cast<int>(next_embedding_regions.size());
    next_coverage.total_samples = 0;

    std::vector<int> counts;
    std::map<std::string, int> domain_samples;

    for (const auto& r : next_embedding_regions) {
      counts.push_back(r.sample_count);
      next_coverage.total_samples += r.sample_count;
      domain_samples[r.domain] += r.sample_count;
    }

    // Calculate coverage score (based on coefficient of variation)
    float avg = static_cast<float>(next_coverage.total_samples) /
                static_cast<float>(next_coverage.num_regions);
    float variance = 0.0f;
    for (int c : counts) {
      float diff = static_cast<float>(c) - avg;
      variance += diff * diff;
    }
    variance /= static_cast<float>(counts.size());
    float std_dev = std::sqrt(variance);
    float cv = avg > 0 ? std_dev / avg : 1.0f;
    next_coverage.coverage_score = std::max(0.0f, std::min(1.0f, 1.0f - cv));

    // Count sparse regions
    float threshold = avg * kSparseCoverageFactor;
    for (int c : counts) {
      if (static_cast<float>(c) < threshold) next_coverage.sparse_regions++;
    }

    // Domain coverage
    for (const auto& [domain, samples] : domain_samples) {
      next_coverage.domain_coverage[domain] =
          static_cast<float>(samples) /
          static_cast<float>(next_coverage.total_samples);
    }
  }

  if (embedding_regions) {
    *embedding_regions = std::move(next_embedding_regions);
  }
  if (coverage) *coverage = std::move(next_coverage);

  result.ok = true;
  return result;
}

DataLoader::LoadResult DataLoader::LoadTrainingFeedback(
    std::vector<TrainingRunData>* training_runs,
    OptimizationData* optimization_data) {
  LoadResult result;
  std::string path = data_path_ + "/training_feedback.json";
  if (!path_exists_(path)) return result;

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty()
                       ? "training_feedback.json is empty"
                       : ("training_feedback.json: " + read_error);
    return result;
  }

  std::string parse_error;
  JsonValue data = ParseJson(content, &parse_error);
  if (!parse_error.empty()) {
    result.ok = false;
    result.error = "training_feedback.json: " + parse_error;
    return result;
  }
  if (!data.IsObject()) {
    result.ok = false;
    result.error = "training_feedback.json: invalid JSON root object";
    return result;
  }

  std::vector<TrainingRunData> next_training_runs;
  OptimizationData next_optimization_data;

  // Parse training runs
  const auto& runs = data["training_runs"];
  if (runs.IsObject()) {
    for (const auto& [run_id, run_data] : runs.object_value) {
      if (!run_data["final_loss"].IsNull()) {
        TrainingRunData trd;
        trd.run_id = run_id;
        trd.model_name = run_data["model_name"].GetString();
        trd.base_model = run_data["base_model"].GetString();
        trd.dataset_path = run_data["dataset_path"].GetString();
        trd.start_time = run_data["start_time"].GetString();
        trd.end_time = run_data["end_time"].GetString();
        trd.notes = run_data["notes"].GetString();
        trd.final_loss = run_data["final_loss"].GetFloat();
        trd.samples_count = run_data["samples_count"].GetInt();

        const auto& dist = run_data["domain_distribution"];
        if (dist.IsObject()) {
          for (const auto& [domain, count] : dist.object_value) {
            trd.domain_distribution[domain] = count.GetInt();
          }
        }

        const auto& metrics = run_data["eval_metrics"];
        if (metrics.IsObject()) {
          for (const auto& [metric, value] : metrics.object_value) {
            trd.eval_metrics[metric] = value.GetFloat();
          }
        }

        next_training_runs.push_back(std::move(trd));
      }
    }
  }

  // Parse domain effectiveness
  const auto& effectiveness = data["domain_effectiveness"];
  if (effectiveness.IsObject()) {
    for (const auto& [domain, value] : effectiveness.object_value) {
      next_optimization_data.domain_effectiveness[domain] = value.GetFloat();
    }
  }

  // Parse threshold sensitivity
  const auto& sensitivity = data["quality_threshold_effectiveness"];
  if (sensitivity.IsObject()) {
    for (const auto& [threshold, value] : sensitivity.object_value) {
      next_optimization_data.threshold_sensitivity[threshold] = value.GetFloat();
    }
  }

  if (training_runs) *training_runs = std::move(next_training_runs);
  if (optimization_data) *optimization_data = std::move(next_optimization_data);

  result.ok = true;
  return result;
}

}  // namespace viz
}  // namespace hafs
