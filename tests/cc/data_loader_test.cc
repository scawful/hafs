#include "data_loader.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"

namespace hafs {
namespace viz {
namespace {

class TempDir {
 public:
  TempDir() {
    std::filesystem::path base = std::filesystem::temp_directory_path();
    auto suffix = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    path_ = base / std::filesystem::path("hafs_viz_test_" + suffix);
    std::filesystem::create_directories(path_);
  }

  ~TempDir() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

  std::filesystem::path File(const std::string& name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

void WriteFile(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream file(path, std::ios::trunc);
  file << contents;
}

TEST(DataLoaderTest, LoadsQualityFeedback) {
  TempDir temp_dir;
  WriteFile(temp_dir.File("quality_feedback.json"), R"json(
{
  "generator_stats": {
    "gen_one": {
      "samples_generated": 100,
      "samples_accepted": 60,
      "samples_rejected": 40,
      "avg_quality_score": 0.8,
      "rejection_reasons": {
        "too_long": 10,
        "too_short": 30
      }
    }
  },
  "rejection_history": [
    {"domain": "math", "scores": {"accuracy": 0.7}},
    {"domain": "math", "scores": {"accuracy": 0.8}},
    {"domain": "math", "scores": {"accuracy": 0.9}},
    {"domain": "math", "scores": {"accuracy": 0.85}},
    {"domain": "math", "scores": {"accuracy": 0.88}}
  ]
}
)json");

  DataLoader loader(temp_dir.path().string());
  ASSERT_TRUE(loader.Refresh());
  ASSERT_TRUE(loader.HasData());

  const auto& stats = loader.GetGeneratorStats();
  ASSERT_EQ(stats.size(), 1u);
  EXPECT_EQ(stats[0].name, "gen_one");
  EXPECT_EQ(stats[0].samples_generated, 100);
  EXPECT_EQ(stats[0].samples_accepted, 60);
  EXPECT_EQ(stats[0].samples_rejected, 40);

  const auto& rejection = loader.GetRejectionSummary();
  EXPECT_EQ(rejection.total_rejections, 40);

  const auto& trends = loader.GetQualityTrends();
  ASSERT_EQ(trends.size(), 1u);
  EXPECT_EQ(trends[0].domain, "math");
  EXPECT_EQ(trends[0].metric, "accuracy");
  EXPECT_EQ(trends[0].trend_direction, "improving");
}

TEST(DataLoaderTest, LoadsActiveLearningAndCoverage) {
  TempDir temp_dir;
  WriteFile(temp_dir.File("active_learning.json"), R"json(
{
  "regions": [
    {"sample_count": 5, "domain": "alpha", "avg_quality": 0.7},
    {"sample_count": 15, "domain": "alpha", "avg_quality": 0.8},
    {"sample_count": 20, "domain": "beta", "avg_quality": 0.6}
  ]
}
)json");

  DataLoader loader(temp_dir.path().string());
  ASSERT_TRUE(loader.Refresh());
  const auto& regions = loader.GetEmbeddingRegions();
  ASSERT_EQ(regions.size(), 3u);

  const auto& coverage = loader.GetCoverage();
  EXPECT_EQ(coverage.total_samples, 40);
  EXPECT_EQ(coverage.num_regions, 3);
  EXPECT_EQ(coverage.domain_coverage.at("alpha"), 0.5f);
  EXPECT_EQ(coverage.domain_coverage.at("beta"), 0.5f);
  EXPECT_EQ(coverage.sparse_regions, 1);
}

TEST(DataLoaderTest, LoadsTrainingFeedbackAndOptimization) {
  TempDir temp_dir;
  WriteFile(temp_dir.File("training_feedback.json"), R"json(
{
  "training_runs": {
    "run-01": {
      "model_name": "m1",
      "base_model": "base",
      "dataset_path": "/tmp/data",
      "start_time": "2024-01-01T00:00:00",
      "end_time": "2024-01-01T01:00:00",
      "notes": "ok",
      "final_loss": 0.12,
      "samples_count": 500,
      "domain_distribution": {"math": 200, "code": 300},
      "eval_metrics": {"accuracy": 0.9}
    }
  },
  "domain_effectiveness": {"math": 0.7, "code": 0.8},
  "quality_threshold_effectiveness": {"0.7": 0.6, "0.8": 0.5}
}
)json");

  DataLoader loader(temp_dir.path().string());
  ASSERT_TRUE(loader.Refresh());

  const auto& runs = loader.GetTrainingRuns();
  ASSERT_EQ(runs.size(), 1u);
  EXPECT_EQ(runs[0].run_id, "run-01");
  EXPECT_EQ(runs[0].samples_count, 500);

  const auto& optimization = loader.GetOptimizationData();
  EXPECT_EQ(optimization.domain_effectiveness.at("math"), 0.7f);
  EXPECT_EQ(optimization.threshold_sensitivity.at("0.8"), 0.5f);
}

TEST(DataLoaderTest, RetainsPreviousDataOnInvalidJson) {
  TempDir temp_dir;
  WriteFile(temp_dir.File("quality_feedback.json"), R"json(
{
  "generator_stats": {},
  "rejection_history": [
    {"domain": "math", "scores": {"accuracy": 0.7}},
    {"domain": "math", "scores": {"accuracy": 0.8}},
    {"domain": "math", "scores": {"accuracy": 0.9}},
    {"domain": "math", "scores": {"accuracy": 0.85}},
    {"domain": "math", "scores": {"accuracy": 0.88}}
  ]
}
)json");

  DataLoader loader(temp_dir.path().string());
  ASSERT_TRUE(loader.Refresh());
  const auto& initial = loader.GetQualityTrends();
  ASSERT_EQ(initial.size(), 1u);

  WriteFile(temp_dir.File("quality_feedback.json"), "{broken");
  ASSERT_FALSE(loader.Refresh());
  const auto& after = loader.GetQualityTrends();
  EXPECT_EQ(after.size(), 1u);
  EXPECT_GT(loader.GetLastStatus().error_count, 0);
}

TEST(DataLoaderTest, UsesAbslForPaths) {
  TempDir temp_dir;
  std::string file_name = absl::StrCat("quality_", "feedback.json");
  WriteFile(temp_dir.File(file_name), R"json({"generator_stats": {}})json");

  DataLoader loader(temp_dir.path().string());
  EXPECT_TRUE(loader.Refresh());
}

}  // namespace
}  // namespace viz
}  // namespace hafs
