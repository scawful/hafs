#include "data_loader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

// Use nlohmann/json for JSON parsing (simpler than simdjson for this use case)
// If simdjson is available, we could optimize later
#include <filesystem>

namespace fs = std::filesystem;

namespace hafs {
namespace viz {

namespace {

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

// Forward declarations for recursive parsing
JsonValue ParseValue(const std::string& json, size_t& pos);

void SkipWhitespace(const std::string& json, size_t& pos) {
  while (pos < json.size() && std::isspace(json[pos])) ++pos;
}

std::string ParseString(const std::string& json, size_t& pos) {
  if (pos >= json.size() || json[pos] != '"') return "";
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

  if (pos < json.size()) ++pos;  // skip closing quote
  return result;
}

double ParseNumber(const std::string& json, size_t& pos) {
  size_t start = pos;
  if (json[pos] == '-') ++pos;
  while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' ||
                               json[pos] == 'e' || json[pos] == 'E' ||
                               json[pos] == '+' || json[pos] == '-')) {
    if ((json[pos] == 'e' || json[pos] == 'E') && pos > start) {
      ++pos;
      if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) ++pos;
    } else {
      ++pos;
    }
  }
  return std::stod(json.substr(start, pos - start));
}

JsonValue ParseArray(const std::string& json, size_t& pos) {
  JsonValue result;
  result.type = JsonValue::Type::Array;

  ++pos;  // skip '['
  SkipWhitespace(json, pos);

  while (pos < json.size() && json[pos] != ']') {
    result.array_value.push_back(ParseValue(json, pos));
    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == ',') {
      ++pos;
      SkipWhitespace(json, pos);
    }
  }

  if (pos < json.size()) ++pos;  // skip ']'
  return result;
}

JsonValue ParseObject(const std::string& json, size_t& pos) {
  JsonValue result;
  result.type = JsonValue::Type::Object;

  ++pos;  // skip '{'
  SkipWhitespace(json, pos);

  while (pos < json.size() && json[pos] != '}') {
    std::string key = ParseString(json, pos);
    SkipWhitespace(json, pos);

    if (pos < json.size() && json[pos] == ':') {
      ++pos;
      SkipWhitespace(json, pos);
    }

    result.object_value[key] = ParseValue(json, pos);
    SkipWhitespace(json, pos);

    if (pos < json.size() && json[pos] == ',') {
      ++pos;
      SkipWhitespace(json, pos);
    }
  }

  if (pos < json.size()) ++pos;  // skip '}'
  return result;
}

JsonValue ParseValue(const std::string& json, size_t& pos) {
  SkipWhitespace(json, pos);

  if (pos >= json.size()) return JsonValue{};

  char c = json[pos];

  if (c == '{') {
    return ParseObject(json, pos);
  } else if (c == '[') {
    return ParseArray(json, pos);
  } else if (c == '"') {
    JsonValue v;
    v.type = JsonValue::Type::String;
    v.string_value = ParseString(json, pos);
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
    v.number_value = ParseNumber(json, pos);
    return v;
  }

  return JsonValue{};
}

JsonValue ParseJson(const std::string& json) {
  size_t pos = 0;
  return ParseValue(json, pos);
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return "";

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace

DataLoader::DataLoader(const std::string& data_path) : data_path_(data_path) {}

bool DataLoader::Refresh() {
  has_data_ = false;
  last_error_.clear();

  quality_trends_.clear();
  generator_stats_.clear();
  embedding_regions_.clear();
  training_runs_.clear();
  coverage_ = CoverageData{};
  rejection_summary_ = RejectionSummary{};
  optimization_data_ = OptimizationData{};

  if (!fs::exists(data_path_)) {
    last_error_ = "Data path does not exist: " + data_path_;
    return false;
  }

  bool quality_ok = LoadQualityFeedback();
  bool active_ok = LoadActiveLearning();
  bool training_ok = LoadTrainingFeedback();

  has_data_ = !generator_stats_.empty() || !embedding_regions_.empty() ||
              !training_runs_.empty();

  return quality_ok || active_ok || training_ok;
}

bool DataLoader::LoadQualityFeedback() {
  std::string path = data_path_ + "/quality_feedback.json";
  if (!fs::exists(path)) return true;  // Optional file

  std::string content = ReadFile(path);
  if (content.empty()) {
    last_error_ = "Failed to read quality_feedback.json";
    return false;
  }

  JsonValue data = ParseJson(content);
  if (!data.IsObject()) {
    last_error_ = "Invalid JSON in quality_feedback.json";
    return false;
  }

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
          rejection_summary_.reasons[reason] += c;
          rejection_summary_.total_rejections += c;
        }
      }

      generator_stats_.push_back(std::move(gs));
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
        if (trend.values.size() < 5) {
          trend.trend_direction = "insufficient";
        } else {
          float recent = 0.0f, older = 0.0f;
          for (size_t i = trend.values.size() - 5; i < trend.values.size(); ++i)
            recent += trend.values[i];
          for (size_t i = 0; i < 5; ++i) older += trend.values[i];
          recent /= 5.0f;
          older /= 5.0f;
          float diff = recent - older;

          if (diff > 0.05f)
            trend.trend_direction = "improving";
          else if (diff < -0.05f)
            trend.trend_direction = "declining";
          else
            trend.trend_direction = "stable";
        }
      }
      quality_trends_.push_back(std::move(trend));
    }
  }

  return true;
}

bool DataLoader::LoadActiveLearning() {
  std::string path = data_path_ + "/active_learning.json";
  if (!fs::exists(path)) return true;  // Optional file

  std::string content = ReadFile(path);
  if (content.empty()) {
    last_error_ = "Failed to read active_learning.json";
    return false;
  }

  JsonValue data = ParseJson(content);
  if (!data.IsObject()) {
    last_error_ = "Invalid JSON in active_learning.json";
    return false;
  }

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
      embedding_regions_.push_back(std::move(erd));
    }
  }

  // Calculate coverage stats
  if (!embedding_regions_.empty()) {
    coverage_.num_regions = static_cast<int>(embedding_regions_.size());
    coverage_.total_samples = 0;

    std::vector<int> counts;
    std::map<std::string, int> domain_samples;

    for (const auto& r : embedding_regions_) {
      counts.push_back(r.sample_count);
      coverage_.total_samples += r.sample_count;
      domain_samples[r.domain] += r.sample_count;
    }

    // Calculate coverage score (based on coefficient of variation)
    float avg = static_cast<float>(coverage_.total_samples) /
                static_cast<float>(coverage_.num_regions);
    float variance = 0.0f;
    for (int c : counts) {
      float diff = static_cast<float>(c) - avg;
      variance += diff * diff;
    }
    variance /= static_cast<float>(counts.size());
    float std_dev = std::sqrt(variance);
    float cv = avg > 0 ? std_dev / avg : 1.0f;
    coverage_.coverage_score = std::max(0.0f, std::min(1.0f, 1.0f - cv));

    // Count sparse regions
    float threshold = avg * 0.5f;
    for (int c : counts) {
      if (static_cast<float>(c) < threshold) coverage_.sparse_regions++;
    }

    // Domain coverage
    for (const auto& [domain, samples] : domain_samples) {
      coverage_.domain_coverage[domain] =
          static_cast<float>(samples) /
          static_cast<float>(coverage_.total_samples);
    }
  }

  return true;
}

bool DataLoader::LoadTrainingFeedback() {
  std::string path = data_path_ + "/training_feedback.json";
  if (!fs::exists(path)) return true;  // Optional file

  std::string content = ReadFile(path);
  if (content.empty()) {
    last_error_ = "Failed to read training_feedback.json";
    return false;
  }

  JsonValue data = ParseJson(content);
  if (!data.IsObject()) {
    last_error_ = "Invalid JSON in training_feedback.json";
    return false;
  }

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

        training_runs_.push_back(std::move(trd));
      }
    }
  }

  // Parse domain effectiveness
  const auto& effectiveness = data["domain_effectiveness"];
  if (effectiveness.IsObject()) {
    for (const auto& [domain, value] : effectiveness.object_value) {
      optimization_data_.domain_effectiveness[domain] = value.GetFloat();
    }
  }

  // Parse threshold sensitivity
  const auto& sensitivity = data["quality_threshold_effectiveness"];
  if (sensitivity.IsObject()) {
    for (const auto& [threshold, value] : sensitivity.object_value) {
      optimization_data_.threshold_sensitivity[threshold] = value.GetFloat();
    }
  }

  return true;
}

}  // namespace viz
}  // namespace hafs
