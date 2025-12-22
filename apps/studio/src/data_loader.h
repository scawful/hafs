#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hafs {
namespace viz {

/// Quality trend for a single domain/metric.
struct QualityTrendData {
  std::string domain;
  std::string metric;
  std::vector<float> values;
  float mean = 0.0f;
  std::string trend_direction;  // "improving", "declining", "stable", "insufficient"
};

/// Statistics for a single generator.
struct GeneratorStatsData {
  std::string name;
  int samples_generated = 0;
  int samples_accepted = 0;
  int samples_rejected = 0;
  std::map<std::string, int> rejection_reasons;
  float acceptance_rate = 0.0f;
  float avg_quality = 0.0f;
};

/// Embedding space region data.
struct EmbeddingRegionData {
  int index = 0;
  int sample_count = 0;
  std::string domain;
  float avg_quality = 0.0f;
};

/// Training run metadata.
struct TrainingRunData {
  std::string run_id;
  std::string model_name;
  std::string base_model;
  std::string dataset_path;
  std::string start_time;
  std::string end_time;
  std::string notes;
  float final_loss = 0.0f;
  int samples_count = 0;
  std::map<std::string, int> domain_distribution;
  std::map<std::string, float> eval_metrics;
};

/// Embedding coverage summary.
struct CoverageData {
  int total_samples = 0;
  int num_regions = 0;
  float coverage_score = 0.0f;
  int sparse_regions = 0;
  std::map<std::string, float> domain_coverage;
};

/// Aggregated rejection reasons across all generators.
struct RejectionSummary {
  std::map<std::string, int> reasons;
  int total_rejections = 0;
};

/// Optimization metrics.
struct OptimizationData {
  std::map<std::string, float> domain_effectiveness;
  std::map<std::string, float> threshold_sensitivity;
};

/// Data load status for the last refresh cycle.
struct LoadStatus {
  bool quality_found = false;
  bool quality_ok = false;
  bool active_found = false;
  bool active_ok = false;
  bool training_found = false;
  bool training_ok = false;
  int error_count = 0;
  std::string last_error;
  std::string last_error_source;

  int FoundCount() const {
    return static_cast<int>(quality_found) + static_cast<int>(active_found) +
           static_cast<int>(training_found);
  }
  int OkCount() const {
    return static_cast<int>(quality_ok) + static_cast<int>(active_ok) +
           static_cast<int>(training_ok);
  }
  bool AnyOk() const { return quality_ok || active_ok || training_ok; }
};

/// Loads training data from JSON files.
class DataLoader {
 public:
  using FileReader = std::function<bool(const std::string&,
                                        std::string*,
                                        std::string*)>;
  using PathExists = std::function<bool(const std::string&)>;

  explicit DataLoader(const std::string& data_path,
                      FileReader file_reader = {},
                      PathExists path_exists = {});

  /// Reload all data from disk. Returns true on success.
  bool Refresh();

  // Accessors
  const std::vector<QualityTrendData>& GetQualityTrends() const {
    return quality_trends_;
  }
  const std::vector<GeneratorStatsData>& GetGeneratorStats() const {
    return generator_stats_;
  }
  const std::vector<EmbeddingRegionData>& GetEmbeddingRegions() const {
    return embedding_regions_;
  }
  const std::vector<TrainingRunData>& GetTrainingRuns() const {
    return training_runs_;
  }
  const CoverageData& GetCoverage() const { return coverage_; }
  const RejectionSummary& GetRejectionSummary() const {
    return rejection_summary_;
  }
  const OptimizationData& GetOptimizationData() const {
    return optimization_data_;
  }
  const LoadStatus& GetLastStatus() const { return last_status_; }

  bool HasData() const { return has_data_; }
  std::string GetLastError() const { return last_error_; }

 private:
  struct LoadResult {
    bool found = false;
    bool ok = false;
    std::string error;
  };

  LoadResult LoadQualityFeedback(std::vector<QualityTrendData>* quality_trends,
                                 std::vector<GeneratorStatsData>* generator_stats,
                                 RejectionSummary* rejection_summary);
  LoadResult LoadActiveLearning(
      std::vector<EmbeddingRegionData>* embedding_regions,
      CoverageData* coverage);
  LoadResult LoadTrainingFeedback(std::vector<TrainingRunData>* training_runs,
                                  OptimizationData* optimization_data);

  std::string data_path_;
  FileReader file_reader_;
  PathExists path_exists_;
  bool has_data_ = false;
  std::string last_error_;
  LoadStatus last_status_;

  std::vector<QualityTrendData> quality_trends_;
  std::vector<GeneratorStatsData> generator_stats_;
  std::vector<EmbeddingRegionData> embedding_regions_;
  std::vector<TrainingRunData> training_runs_;
  CoverageData coverage_;
  RejectionSummary rejection_summary_;
  OptimizationData optimization_data_;
};

}  // namespace viz
}  // namespace hafs
