#pragma once

#include <array>
#include <deque>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>
#include "data_loader.h"
#include "widgets/text_editor.h"
#include "widgets/imgui_memory_editor.h"

// Forward declarations - implot.h included in .cc
struct ImPlotContext;
struct GLFWwindow;

namespace hafs {
namespace viz {

enum class Workspace { Dashboard, Analysis, Optimization, Systems, Custom, Chat, Training, Context };
enum class ThemeProfile { Cobalt, Amber, Emerald };

enum class PlotKind {
  None,
  QualityTrends,
  GeneratorEfficiency,
  CoverageDensity,
  TrainingLoss,
  LossVsSamples,
  DomainCoverage,
  EmbeddingQuality,
  AgentThroughput,
  MissionQueue,
  QualityDirection,
  GeneratorMix,
  EmbeddingDensity,
  AgentUtilization,
  MissionProgress,
  EvalMetrics,
  Rejections,
  KnowledgeGraph,
  LatentSpace,
  Effectiveness,
  Thresholds,
};

struct MetricCard {
  std::string label;
  std::string value;
  std::string sub_text;
  ImVec4 color;
  bool positive_trend = true;
};

struct AgentState {
  std::string name;
  std::string role;
  std::string status;
  bool enabled = true;
  bool data_backed = false;
  int queue_depth = 0;
  int tasks_completed = 0;
  float success_rate = 0.0f;
  float avg_latency_ms = 0.0f;
  float cpu_pct = 0.0f;
  float mem_pct = 0.0f;
  float activity_phase = 0.0f;
};

struct MissionState {
  std::string name;
  std::string owner;
  std::string status;
  bool data_backed = false;
  int priority = 2;
  float progress = 0.0f;
};

struct LogEntry {
  std::string agent;
  std::string message;
  std::string kind;
};

struct FileEntry {
  std::string name;
  std::filesystem::path path;
  bool is_directory;
  uintmax_t size;
  bool has_context = false;
};

struct ContextItem {
  std::string name;
  std::filesystem::path path;
  std::string type;
  bool enabled = true;
};

/// Main visualization application.
class App {
 public:
  explicit App(const std::string& data_path);
  ~App();

  /// Run the application main loop.
  int Run();

  /// Get the data loader for accessing training data.
  DataLoader& GetLoader() { return loader_; }
  const DataLoader& GetLoader() const { return loader_; }

 private:
  bool InitWindow();
  bool InitImGui();
  void Shutdown();

  void RefreshData(const char* reason);
  void MaybeAutoRefresh();
  void SeedDefaultState();
  void SyncDataBackedState();
  void TickSimulatedMetrics(float dt);
  void AppendLog(const std::string& agent, const std::string& message,
                 const std::string& kind);

  void RenderFrame();
  void RenderMenuBar();
  void RenderDockSpace();
  void RenderLayout();
  void RenderSummaryRow();
  void RenderControlColumn();
  void RenderMetricsColumn();
  void RenderKnobsTab();
  void RenderAgentsTab();
  void RenderMissionsTab();
  void RenderLogsTab();
  void RenderQualityChart();
  void RenderGeneratorChart();
  void RenderCoverageChart();
  void RenderTrainingChart();
  void RenderTrainingLossChart();
  void RenderRejectionChart();
  void RenderQualityDirectionChart();
  void RenderGeneratorMixChart();
  void RenderEmbeddingDensityChart();
  void RenderAgentUtilizationChart();
  void RenderMissionProgressChart();
  void RenderEvalMetricsChart();
  void RenderEffectivenessChart();
  void RenderThresholdOptimizationChart();
  void RenderDomainCoverageChart();
  void RenderEmbeddingQualityChart();
  void RenderAgentThroughputChart();
  void RenderMissionQueueChart();
  void RenderServicesTab();
  void RenderContextTab();
  void RenderAgentPromptTab();
  void RenderTablesTab();
  void RenderKnowledgeGraph();
  void RenderLatentSpaceChart();
  void RenderStatusBar();
  void RenderInspectorPanel();
  void RenderDatasetPanel();
  void RenderSystemsPanel();

  // Helpers
  void RefreshBrowserEntries();
  void LoadFile(const std::filesystem::path& path);

  // Workspace Views (Consolidated Grids)
  void RenderDashboardView();
  void RenderAnalysisView();
  void RenderOptimizationView();
  void RenderSystemsView();
  void RenderCustomGridView();
  void RenderChatView();
  void RenderTrainingView();
  void RenderContextView();
  void RenderMarkdown(const std::string& content);
  void RenderExpandedPlot();
  void RenderPlotByKind(PlotKind kind);

  // Premium UI Components
  void RenderSidebar();
  void RenderMetricCards();
  void ApplyPremiumPlotStyles(const char* plot_id);
  void HelpMarker(const char* desc);
  void RenderChartHeader(PlotKind kind, const char* title, const char* desc);
  void HandlePlotContextMenu(PlotKind kind);
  ImVec4 GetThemeColor(ImGuiCol col);
  ImVec4 GetSeriesColor(int index);
  ImVec4 GetStepColor(float step); // For pulses
  int GetPlotAxisFlags() const;

  std::string data_path_;
  DataLoader loader_;

  GLFWwindow* window_ = nullptr;
  ImGuiContext* imgui_ctx_ = nullptr;
  ImPlotContext* implot_ctx_ = nullptr;

  int window_width_ = 1400;
  int window_height_ = 900;
  bool should_refresh_ = false;
  bool show_demo_window_ = false;
  bool auto_refresh_ = false;
  bool simulate_activity_ = true;
  bool verbose_logs_ = false;
  bool compact_charts_ = false;
  bool show_status_strip_ = true;
  bool show_controls_ = true;
  bool show_inspector_ = true;
  bool show_dataset_panel_ = true;
  bool show_systems_panel_ = true;
  bool enable_viewports_ = true;
  bool enable_docking_ = true;
  bool reset_layout_on_workspace_change_ = false;
  bool allow_workspace_scroll_ = false;
  bool enable_plot_interaction_ = true;
  bool plot_interaction_requires_modifier_ = true;
  bool auto_chart_columns_ = true;
  bool show_agent_details_ = true;
  bool show_knowledge_graph_ = false;
  float refresh_interval_sec_ = 8.0f;
  float chart_height_ = 170.0f;
  float plot_height_ = 170.0f;
  float agent_activity_scale_ = 1.0f;
  float embedding_sample_rate_ = 0.6f;
  float quality_threshold_ = 0.7f;
  float mission_priority_bias_ = 1.0f;
  int mission_concurrency_ = 4;
  int chart_columns_ = 3;
  int spawn_agent_count_ = 1;
  int spawn_mission_count_ = 1;
  int new_mission_priority_ = 3;
  int log_agent_index_ = 0;
  int selected_agent_index_ = -1;
  int selected_run_index_ = -1;
  int selected_generator_index_ = -1;
  Workspace current_workspace_ = Workspace::Dashboard;
  ThemeProfile current_theme_ = ThemeProfile::Cobalt;
  bool force_reset_layout_ = false;
  bool lock_layout_ = false;
  double last_refresh_time_ = 0.0;

  std::vector<AgentState> agents_;
  std::vector<MissionState> missions_;
  std::deque<LogEntry> logs_;

  std::vector<float> sparkline_data_;  // For shared use

  std::array<char, 64> new_agent_name_{};
  std::array<char, 64> new_agent_role_{};
  std::array<char, 96> new_mission_name_{};
  std::array<char, 64> new_mission_owner_{};
  std::array<char, 128> log_filter_{};
  std::array<char, 96> run_filter_{};
  std::array<char, 96> generator_filter_{};
  std::array<char, 256> chat_input_{};
  std::array<char, 1024> system_prompt_{};
  std::array<char, 1024> user_prompt_{};

  // Context Browser State
  std::filesystem::path current_browser_path_;
  std::vector<FileEntry> browser_entries_;
  std::vector<ContextItem> selected_context_;
  std::string context_filter_;
  std::filesystem::path selected_file_path_;
  bool show_hidden_files_ = false;

  // Editors
  TextEditor text_editor_;
  MemoryEditorWidget memory_editor_;
  std::vector<uint8_t> binary_data_;
  bool is_binary_view_ = false;

  // UI State
  bool show_advanced_tables_ = true;
  bool show_sparklines_ = true;
  bool use_pulse_animations_ = true;
  bool show_plot_legends_ = true;
  bool show_plot_markers_ = true;
  bool data_scientist_mode_ = false;
  bool show_all_charts_ = true;
  float pulse_timer_ = 0.0f;
  int custom_grid_rows_ = 2;
  int custom_grid_columns_ = 2;
  PlotKind expanded_plot_ = PlotKind::None;
  bool is_rendering_expanded_plot_ = false;
  std::vector<PlotKind> custom_grid_slots_;

  // Knowledge Graph State
  std::vector<std::string> knowledge_concepts_;
  std::vector<float> knowledge_nodes_x_;
  std::vector<float> knowledge_nodes_y_;
  struct Edge { int from, to; };
  std::vector<Edge> knowledge_edges_;

  // Typography
  ImFont* font_ui_ = nullptr;
  ImFont* font_header_ = nullptr;
  ImFont* font_mono_ = nullptr; // For code editors
  ImFont* font_icons_ = nullptr;
};

}  // namespace viz
}  // namespace hafs
