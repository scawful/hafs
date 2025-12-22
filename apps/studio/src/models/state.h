#pragma once

#include <string>
#include <vector>
#include <deque>
#include <filesystem>
#include <array>
#include <imgui.h>

namespace hafs {
namespace viz {

enum class Workspace { Dashboard, Analysis, Optimization, Systems, Custom, Chat, Training, Context, Models };
enum class ThemeProfile { Cobalt, Amber, Emerald, Cyberpunk, Monochrome, Solarized, Nord, Dracula, Default = Cobalt };

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

struct AppState {
  // App-level flags
  bool should_refresh = false;
  bool show_demo_window = false;
  bool auto_refresh = false;
  bool simulate_activity = true;
  bool verbose_logs = false;
  bool compact_charts = false;
  bool show_status_strip = true;
  bool show_controls = true;
  bool show_inspector = true;
  bool show_dataset_panel = true;
  bool show_systems_panel = true;
  bool enable_viewports = true;
  bool enable_docking = true;
  bool reset_layout_on_workspace_change = false;
  bool allow_workspace_scroll = false;
  bool enable_plot_interaction = true;
  bool plot_interaction_requires_modifier = true;
  bool auto_chart_columns = true;
  bool show_agent_details = true;
  bool show_knowledge_graph = false;
  
  // Visual/Grid Config
  float refresh_interval_sec = 8.0f;
  float chart_height = 170.0f;
  float plot_height = 170.0f;
  float agent_activity_scale = 1.0f;
  float embedding_sample_rate = 0.6f;
  float quality_threshold = 0.7f;
  float mission_priority_bias = 1.0f;
  int mission_concurrency = 4;
  int chart_columns = 3;
  int spawn_agent_count = 1;
  int spawn_mission_count = 1;
  int new_mission_priority = 3;
  int log_agent_index = 0;
  int selected_agent_index = -1;
  int selected_run_index = -1;
  std::string selected_run_id;
  std::vector<std::string> compared_run_ids;
  int selected_generator_index = -1;
  std::string selected_generator_name;
  Workspace current_workspace = Workspace::Dashboard;
  ThemeProfile current_theme = ThemeProfile::Cobalt;
  bool force_reset_layout = false;
  bool lock_layout = false;
  double last_refresh_time = 0.0;

  // Data Collections
  std::vector<AgentState> agents;
  std::vector<MissionState> missions;
  std::deque<LogEntry> logs;
  std::vector<float> sparkline_data;

  // Input Buffers
  std::array<char, 64> new_agent_name{};
  std::array<char, 64> new_agent_role{};
  std::array<char, 96> new_mission_name{};
  std::array<char, 64> new_mission_owner{};
  std::array<char, 128> log_filter{};
  std::array<char, 96> run_filter{};
  std::array<char, 96> generator_filter{};
  std::array<char, 256> chat_input{};
  std::array<char, 1024> system_prompt{};
  std::array<char, 1024> user_prompt{};

  // Trainer Config
  float trainer_lr = 0.0005f;
  int trainer_epochs = 10;
  int trainer_batch_size = 32;
  float generator_temp = 0.7f;
  float rejection_threshold = 0.65f;

  // Context Browser
  std::filesystem::path current_browser_path;
  std::vector<FileEntry> browser_entries;
  std::vector<ContextItem> selected_context;
  std::string context_filter;
  std::filesystem::path selected_file_path;
  bool show_hidden_files = false;

  // Editor State
  std::vector<uint8_t> binary_data;
  bool is_binary_view = false;

  // UI State Components
  bool show_advanced_tables = true;
  bool show_sparklines = true;
  bool use_pulse_animations = true;
  bool show_plot_legends = true;
  bool show_plot_markers = true;
  bool data_scientist_mode = false;
  bool show_all_charts = true;
  float pulse_timer = 0.0f;
  int custom_grid_rows = 2;
  int custom_grid_columns = 2;
  PlotKind expanded_plot = PlotKind::None;
  bool is_rendering_expanded_plot = false;
  std::vector<PlotKind> custom_grid_slots;

  // Knowledge Graph
  std::vector<std::string> knowledge_concepts;
  std::vector<float> knowledge_nodes_x;
  std::vector<float> knowledge_nodes_y;
  struct Edge { int from, to; };
  std::vector<Edge> knowledge_edges;
};

} // namespace viz
} // namespace hafs
