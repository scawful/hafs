#include "data_loader.h"
#include "core/logger.h"
#include "core/filesystem.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>
#include <nlohmann/json.hpp>

namespace hafs {
namespace viz {

namespace {

using json = nlohmann::json;

constexpr size_t kTrendWindow = 5;
constexpr float kTrendDeltaThreshold = 0.05f;

bool IsWhitespaceOnly(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isspace(c);
  });
}

} // namespace

DataLoader::DataLoader(const std::string& data_path,
                       FileReader file_reader,
                       PathExists path_exists)
    : data_path_(data_path) {
    
    // Set default handlers if not provided
    if (file_reader) {
        file_reader_ = std::move(file_reader);
    } else {
        file_reader_ = [](const std::string& p, std::string* c, std::string* e) {
            auto content = studio::core::FileSystem::ReadFile(p);
            if (content) {
                *c = *content;
                return true;
            }
            if (e) *e = "Failed to read file";
            return false;
        };
    }

    if (path_exists) {
        path_exists_ = std::move(path_exists);
    } else {
        path_exists_ = [](const std::string& p) {
            return studio::core::FileSystem::Exists(p);
        };
    }
}

bool DataLoader::Refresh() {
  last_error_.clear();
  last_status_ = LoadStatus{};

  if (!path_exists_(data_path_)) {
    last_error_ = "Data path does not exist: " + data_path_;
    LOG_ERROR(last_error_);
    last_status_.error_count = 1;
    last_status_.last_error = last_error_;
    last_status_.last_error_source = "data_path";
    return false;
  }
  LOG_INFO("DataLoader: Refreshing from " + data_path_);

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

  return last_status_.AnyOk() || (!last_status_.FoundCount() > 0 && has_data_);
}

DataLoader::LoadResult DataLoader::LoadQualityFeedback(
    std::vector<QualityTrendData>* quality_trends,
    std::vector<GeneratorStatsData>* generator_stats,
    RejectionSummary* rejection_summary) {
  
  LoadResult result;
  std::string path = data_path_ + "/quality_feedback.json";
  if (!path_exists_(path)) {
      LOG_WARN("quality_feedback.json not found at " + path);
      return result;
  }
  LOG_INFO("DataLoader: Loading " + path);

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "quality_feedback.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<QualityTrendData> next_quality_trends;
    std::vector<GeneratorStatsData> next_generator_stats;
    RejectionSummary next_rejection_summary;

    if (data.contains("generator_stats") && data["generator_stats"].is_object()) {
      for (auto& [name, stats] : data["generator_stats"].items()) {
        GeneratorStatsData gs;
        gs.name = name;
        gs.samples_generated = stats.value("samples_generated", 0);
        gs.samples_accepted = stats.value("samples_accepted", 0);
        gs.samples_rejected = stats.value("samples_rejected", 0);
        gs.avg_quality = stats.value("avg_quality_score", 0.0f);

        int total = gs.samples_accepted + gs.samples_rejected;
        gs.acceptance_rate = total > 0 ? static_cast<float>(gs.samples_accepted) / total : 0.0f;

        if (stats.contains("rejection_reasons") && stats["rejection_reasons"].is_object()) {
          for (auto& [reason, count] : stats["rejection_reasons"].items()) {
            int c = count.get<int>();
            gs.rejection_reasons[reason] = c;
            next_rejection_summary.reasons[reason] += c;
            next_rejection_summary.total_rejections += c;
          }
        }
        next_generator_stats.push_back(std::move(gs));
      }
    }

    if (data.contains("rejection_history") && data["rejection_history"].is_array()) {
      std::map<std::pair<std::string, std::string>, QualityTrendData> trends_map;
      for (auto& entry : data["rejection_history"]) {
        std::string domain = entry.value("domain", "unknown");
        if (entry.contains("scores") && entry["scores"].is_object()) {
          for (auto& [metric, value] : entry["scores"].items()) {
            auto key = std::make_pair(domain, metric);
            if (trends_map.find(key) == trends_map.end()) {
              trends_map[key] = QualityTrendData{domain, metric};
            }
            trends_map[key].values.push_back(value.get<float>());
          }
        }
      }

      for (auto& [key, trend] : trends_map) {
        if (!trend.values.empty()) {
          float sum = 0.0f;
          for (float v : trend.values) sum += v;
          trend.mean = sum / trend.values.size();

          if (trend.values.size() < kTrendWindow) {
            trend.trend_direction = "insufficient";
          } else {
            float recent = 0.0f, older = 0.0f;
            for (size_t i = trend.values.size() - kTrendWindow; i < trend.values.size(); ++i) recent += trend.values[i];
            for (size_t i = 0; i < kTrendWindow && i < trend.values.size(); ++i) older += trend.values[i];
            recent /= kTrendWindow;
            older /= std::min((size_t)kTrendWindow, trend.values.size());
            float diff = recent - older;
            if (diff > kTrendDeltaThreshold) trend.trend_direction = "improving";
            else if (diff < -kTrendDeltaThreshold) trend.trend_direction = "declining";
            else trend.trend_direction = "stable";
          }
        }
        next_quality_trends.push_back(std::move(trend));
      }
    }

    if (quality_trends) *quality_trends = std::move(next_quality_trends);
    if (generator_stats) *generator_stats = std::move(next_generator_stats);
    if (rejection_summary) *rejection_summary = std::move(next_rejection_summary);

    LOG_INFO("DataLoader: Successfully loaded data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in quality_feedback.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadActiveLearning(
    std::vector<EmbeddingRegionData>* embedding_regions,
    CoverageData* coverage) {
  
  LoadResult result;
  std::string path = data_path_ + "/active_learning.json";
  if (!path_exists_(path)) return result;

  LOG_INFO("DataLoader: Loading " + path);
  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "active_learning.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<EmbeddingRegionData> next_embedding_regions;
    CoverageData next_coverage;

    if (data.contains("regions") && data["regions"].is_array()) {
      int idx = 0;
      for (auto& region : data["regions"]) {
        EmbeddingRegionData erd;
        erd.index = idx++;
        erd.sample_count = region.value("sample_count", 0);
        erd.domain = region.value("domain", "unknown");
        erd.avg_quality = region.value("avg_quality", 0.0f);
        next_embedding_regions.push_back(std::move(erd));
      }
    }

    next_coverage.num_regions = data.value("num_regions", 0);
    
    if (embedding_regions) *embedding_regions = std::move(next_embedding_regions);
    if (coverage) *coverage = std::move(next_coverage);

    LOG_INFO("DataLoader: Successfully loaded active learning data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in active_learning.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadTrainingFeedback(
    std::vector<TrainingRunData>* training_runs,
    OptimizationData* optimization_data) {
  
  LoadResult result;
  std::string path = data_path_ + "/training_feedback.json";
  if (!path_exists_(path)) return result;

  LOG_INFO("DataLoader: Loading " + path);
  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path, &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "training_feedback.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<TrainingRunData> next_training_runs;
    OptimizationData next_optimization_data;

    if (data.contains("training_runs") && data["training_runs"].is_object()) {
      for (auto& [id, run] : data["training_runs"].items()) {
        TrainingRunData trd;
        trd.run_id = id;
        trd.model_name = run.value("model_name", "unknown");
        trd.samples_count = run.value("samples_count", 0);
        trd.final_loss = run.value("final_loss", 0.0f);
        trd.start_time = run.value("start_time", "");
        
        if (run.contains("domain_distribution") && run["domain_distribution"].is_object()) {
           for (auto& [domain, count] : run["domain_distribution"].items()) {
             trd.domain_distribution[domain] = count.get<int>();
           }
        }
        next_training_runs.push_back(std::move(trd));
      }
    }

    if (data.contains("domain_effectiveness") && data["domain_effectiveness"].is_object()) {
      for (auto& [domain, val] : data["domain_effectiveness"].items()) {
        next_optimization_data.domain_effectiveness[domain] = val.get<float>();
      }
    }

    if (data.contains("quality_threshold_effectiveness") && data["quality_threshold_effectiveness"].is_object()) {
      for (auto& [thresh, val] : data["quality_threshold_effectiveness"].items()) {
        next_optimization_data.threshold_sensitivity[thresh] = val.get<float>();
      }
    }

    if (training_runs) *training_runs = std::move(next_training_runs);
    if (optimization_data) *optimization_data = std::move(next_optimization_data);

    LOG_INFO("DataLoader: Successfully loaded training feedback data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in training_feedback.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

} // namespace viz
} // namespace hafs
