#include "app.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <vector>

// GLFW + OpenGL
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "implot.h"
#include "themes/hafs_theme.h"
#include "icons.h"

// Modular Components
#include "ui/core.h"
#include "ui/components/metrics.h"
#include "ui/components/charts.h"
#include "ui/components/tabs.h"
#include "ui/components/panels.h"

namespace hafs {
namespace viz {

namespace {
void GlfwErrorCallback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
} // namespace

App::App(const std::string& data_path)
    : data_path_(data_path), loader_(data_path) {
  std::snprintf(state_.new_agent_role.data(), state_.new_agent_role.size(), "Evaluator");
  std::snprintf(state_.new_mission_owner.data(), state_.new_mission_owner.size(), "Ops");
  std::snprintf(state_.system_prompt.data(), state_.system_prompt.size(), 
                "You are a HAFS data science assistant. Analyze the training trends and suggest optimizations.");
  
  const char* home = std::getenv("HOME");
  state_.current_browser_path = home ? std::filesystem::path(home) : std::filesystem::current_path();
  ui::RefreshBrowserEntries(state_);
  SeedDefaultState();
}

App::~App() { Shutdown(); }

bool App::InitWindow() {
  glfwSetErrorCallback(GlfwErrorCallback);
  if (!glfwInit()) return false;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  window_ = glfwCreateWindow(window_width_, window_height_, "HAFS Training Data Visualization", nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);
  return true;
}

bool App::InitImGui() {
  IMGUI_CHECKVERSION();
  imgui_ctx_ = ImGui::CreateContext();
  implot_ctx_ = ImPlot::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  if (state_.enable_docking) io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  if (state_.enable_viewports) io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Setup fonts
  if (!LoadFonts()) {
      fprintf(stderr, "Warning: Failed to load one or more fonts. Using default.\n");
  }

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 150");
  themes::ApplyHafsTheme();
  shortcut_manager_.LoadFromDisk();
  return true;
}

bool App::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    
    // Find project root by looking for apps/studio
    std::filesystem::path current = std::filesystem::current_path();
    std::filesystem::path font_dir;
    
    // Try a few common locations relative to executable
    std::vector<std::filesystem::path> search_paths = {
        current / "assets" / "font",
        current / "src" / "assets" / "font",
        current / ".." / ".." / ".." / "apps" / "studio" / "src" / "assets" / "font",
        current / ".." / ".." / "apps" / "studio" / "src" / "assets" / "font",
        "/Users/scawful/Code/hafs/apps/studio/src/assets/font"
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path / "Roboto-Medium.ttf")) {
            font_dir = path;
            break;
        }
    }

    if (font_dir.empty()) {
        fprintf(stderr, "Error: Could not find font directory. Searching current: %s\n", current.c_str());
        return false;
    }

    float base_size = 15.0f;
    float header_size = 18.0f;
    float mono_size = 14.0f;

    // Helper to merge icons
    auto MergeIcons = [&](float size) {
        static const ImWchar icons_ranges[] = { (ImWchar)ICON_MIN_MD, (ImWchar)0xFFFF, 0 };
        ImFontConfig icons_config;
        icons_config.MergeMode = true;
        icons_config.PixelSnapH = true;
        icons_config.GlyphMinAdvanceX = size;
        return io.Fonts->AddFontFromFileTTF((font_dir / "MaterialIcons-Regular.ttf").string().c_str(), size, &icons_config, icons_ranges);
    };

    // UI Font
    font_ui_ = io.Fonts->AddFontFromFileTTF((font_dir / "Karla-Regular.ttf").string().c_str(), base_size);
    MergeIcons(base_size);
    
    // Header Font
    font_header_ = io.Fonts->AddFontFromFileTTF((font_dir / "Roboto-Medium.ttf").string().c_str(), header_size);
    MergeIcons(header_size);
    
    // Mono Font
    font_mono_ = io.Fonts->AddFontFromFileTTF((font_dir / "Cousine-Regular.ttf").string().c_str(), mono_size);
    MergeIcons(mono_size);
    
    font_icons_ = font_ui_; // Fallback reference

    // io.Fonts->Build(); // Removed to avoid assertion failure in newer ImGui/Backend combos
    return font_ui_ != nullptr;
}

void App::Shutdown() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext(implot_ctx_);
  ImGui::DestroyContext(imgui_ctx_);
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void App::RefreshData(const char* reason) {
  bool ok = loader_.Refresh();
  state_.last_refresh_time = glfwGetTime();
  SyncDataBackedState();
  
  const auto& status = loader_.GetLastStatus();
  std::string msg;
  if (status.error_count > 0) {
    msg = "Data refreshed with errors (";
    msg += reason;
    msg += "): ";
    msg += status.last_error.empty() ? "see logs" : status.last_error;
  } else if (!status.AnyOk() && status.FoundCount() == 0) {
    msg = "No data sources found (";
    msg += reason;
    msg += ")";
  } else if (!ok) {
    msg = "Data refresh failed (";
    msg += reason;
    msg += ")";
  } else {
    msg = "Data refreshed (";
    msg += reason;
    msg += ")";
  }
  ui::AppendLog(state_, "system", msg, "system");
}

void App::SyncDataBackedState() {
  const auto& coverage = loader_.GetCoverage();
  const auto& trends = loader_.GetQualityTrends();
  const auto& runs = loader_.GetTrainingRuns();
  const auto& generators = loader_.GetGeneratorStats();

  // Sync Agent: Region Indexer
  auto* indexer = ui::FindAgentByName(state_.agents, "Region Indexer");
  if (!indexer) {
    state_.agents.emplace_back();
    indexer = &state_.agents.back();
    indexer->name = "Region Indexer";
    indexer->role = "Librarian";
  }
  indexer->data_backed = true;
  indexer->enabled = true;
  indexer->tasks_completed = coverage.total_samples;
  indexer->queue_depth = coverage.sparse_regions;
  indexer->success_rate = ui::Clamp01(coverage.coverage_score);
  indexer->status = indexer->queue_depth > 0 ? "Busy" : "Idle";

  // Sync Agent: Quality Monitor
  float quality_mean = 0.0f;
  int insufficient = 0;
  for (const auto& trend : trends) {
    quality_mean += trend.mean;
    if (trend.trend_direction == "insufficient") ++insufficient;
  }
  if (!trends.empty()) quality_mean /= static_cast<float>(trends.size());

  auto* evaluator = ui::FindAgentByName(state_.agents, "Quality Monitor");
  if (!evaluator) {
    state_.agents.emplace_back();
    evaluator = &state_.agents.back();
    evaluator->name = "Quality Monitor";
    evaluator->role = "Evaluator";
  }
  evaluator->data_backed = true;
  evaluator->enabled = true;
  evaluator->tasks_completed = static_cast<int>(trends.size());
  evaluator->queue_depth = insufficient;
  evaluator->success_rate = ui::Clamp01(quality_mean);
  evaluator->status = evaluator->queue_depth > 0 ? "Review" : "Idle";

  // Sync Agent: Trainer Coordinator
  float avg_loss = 0.0f;
  for (const auto& run : runs) avg_loss += run.final_loss;
  if (!runs.empty()) avg_loss /= static_cast<float>(runs.size());

  int high_loss = 0;
  for (const auto& run : runs) if (run.final_loss > avg_loss) ++high_loss;

  auto* trainer = ui::FindAgentByName(state_.agents, "Trainer Coordinator");
  if (!trainer) {
    state_.agents.emplace_back();
    trainer = &state_.agents.back();
    trainer->name = "Trainer Coordinator";
    trainer->role = "Trainer";
  }
  trainer->data_backed = true;
  trainer->enabled = true;
  trainer->tasks_completed = static_cast<int>(runs.size());
  trainer->queue_depth = high_loss;
  trainer->success_rate = avg_loss > 0.0f ? ui::Clamp01(1.0f / (1.0f + avg_loss)) : 0.0f;
  trainer->status = trainer->queue_depth > 0 ? "Training" : "Idle";

  // Sync Missions
  state_.missions.erase(std::remove_if(state_.missions.begin(), state_.missions.end(), [](const MissionState& m) { return m.data_backed; }), state_.missions.end());
  for (const auto& run : runs) {
    MissionState mission;
    mission.data_backed = true;
    mission.owner = run.model_name.empty() ? "Trainer" : run.model_name;
    mission.name = run.run_id.size() > 12 ? run.run_id.substr(0, 12) : run.run_id;
    mission.status = "Complete";
    mission.priority = run.final_loss > avg_loss ? 4 : 3;
    mission.progress = 1.0f;
    state_.missions.push_back(std::move(mission));
  }

  if (!state_.selected_run_id.empty()) {
    auto it = std::find_if(runs.begin(), runs.end(),
                           [&](const TrainingRunData& run) {
                             return run.run_id == state_.selected_run_id;
                           });
    if (it != runs.end()) {
      state_.selected_run_index = static_cast<int>(it - runs.begin());
    } else {
      state_.selected_run_index = -1;
      state_.selected_run_id.clear();
    }
  } else if (state_.selected_run_index >= 0 &&
             state_.selected_run_index < static_cast<int>(runs.size())) {
    state_.selected_run_id = runs[state_.selected_run_index].run_id;
  }

  if (!state_.selected_generator_name.empty()) {
    auto it = std::find_if(generators.begin(), generators.end(),
                           [&](const GeneratorStatsData& gen) {
                             return gen.name == state_.selected_generator_name;
                           });
    if (it != generators.end()) {
      state_.selected_generator_index =
          static_cast<int>(it - generators.begin());
    } else {
      state_.selected_generator_index = -1;
      state_.selected_generator_name.clear();
    }
  } else if (state_.selected_generator_index >= 0 &&
             state_.selected_generator_index < static_cast<int>(generators.size())) {
    state_.selected_generator_name =
        generators[state_.selected_generator_index].name;
  }
}

void App::SeedDefaultState() {
  state_.agents.clear();
  state_.missions.clear();
  state_.logs.clear();

  // Create some initial agents and missions
  SyncDataBackedState();
  
  if (state_.agents.size() < 4) {
      AgentState agent;
      agent.name = "Swarm Lead";
      agent.role = "Lead";
      agent.status = "Idle";
      agent.enabled = true;
      agent.success_rate = 0.95f;
      state_.agents.push_back(std::move(agent));
  }

  ui::AppendLog(state_, "system", "HAFS Visualizer initialized.", "system");
}

void App::TickSimulatedMetrics(float dt) {
  state_.pulse_timer += dt;
  if (state_.pulse_timer > 6.28318f) state_.pulse_timer -= 6.28318f;

  if (!state_.simulate_activity) return;

  float time = static_cast<float>(glfwGetTime());
  for (auto& agent : state_.agents) {
    if (agent.data_backed || !agent.enabled) continue;
    float wave = 0.5f + 0.5f * std::sin(time + agent.activity_phase);
    int queue_delta = static_cast<int>(wave * 3.0f * state_.agent_activity_scale) - 1;
    agent.queue_depth = std::max(0, agent.queue_depth + queue_delta);
    agent.tasks_completed += static_cast<int>(wave * 2.0f * state_.agent_activity_scale);
    agent.success_rate = ui::Clamp01(agent.success_rate + (wave - 0.5f) * 0.02f);
    agent.avg_latency_ms = std::max(6.0f, agent.avg_latency_ms + (0.5f - wave) * 2.0f);
    agent.cpu_pct = std::min(95.0f, std::max(5.0f, agent.cpu_pct + (wave - 0.5f) * 6.0f));
    agent.mem_pct = std::min(95.0f, std::max(5.0f, agent.mem_pct + (wave - 0.5f) * 4.0f));
    agent.status = agent.queue_depth > 0 ? "Busy" : "Idle";
  }

  for (auto& mission : state_.missions) {
    if (mission.data_backed || mission.status == "Complete") continue;

    float rate = (0.08f + 0.04f * static_cast<float>(mission.priority)) * state_.mission_priority_bias;
    mission.progress = ui::Clamp01(mission.progress + dt * rate);
    if (mission.progress > 0.02f && mission.status == "Queued") mission.status = "Active";
    if (mission.progress >= 1.0f) {
      mission.status = "Complete";
      ui::AppendLog(state_, "system", "Mission complete: " + mission.name, "system");
    }
  }
}

void App::MaybeAutoRefresh() {
  if (state_.auto_refresh && (glfwGetTime() - state_.last_refresh_time > state_.refresh_interval_sec)) {
    RefreshData("auto");
  }
}

void App::HandleShortcuts() {
  const ImGuiIO& io = ImGui::GetIO();

  if (shortcut_manager_.IsTriggered(ui::ActionId::Refresh, io)) {
    state_.should_refresh = true;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::Quit, io)) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleInspector, io)) {
    state_.show_inspector = !state_.show_inspector;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleDatasetPanel, io)) {
    state_.show_dataset_panel = !state_.show_dataset_panel;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleSystemsPanel, io)) {
    state_.show_systems_panel = !state_.show_systems_panel;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleStatusBar, io)) {
    state_.show_status_strip = !state_.show_status_strip;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleControls, io)) {
    state_.show_controls = !state_.show_controls;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleAutoRefresh, io)) {
    state_.auto_refresh = !state_.auto_refresh;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleSimulation, io)) {
    state_.simulate_activity = !state_.simulate_activity;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleCompactUI, io)) {
    state_.compact_charts = !state_.compact_charts;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleLockLayout, io)) {
    state_.lock_layout = !state_.lock_layout;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ResetLayout, io)) {
    state_.force_reset_layout = true;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleSampleReview, io)) {
    show_sample_review_ = !show_sample_review_;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleShortcutsWindow, io)) {
    show_shortcuts_window_ = !show_shortcuts_window_;
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::ToggleDemoWindow, io)) {
    state_.show_demo_window = !state_.show_demo_window;
  }

  auto set_workspace = [&](Workspace workspace) {
    if (state_.current_workspace == workspace) return;
    state_.current_workspace = workspace;
    if (state_.reset_layout_on_workspace_change) state_.force_reset_layout = true;
  };

  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceDashboard, io)) {
    set_workspace(Workspace::Dashboard);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceAnalysis, io)) {
    set_workspace(Workspace::Analysis);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceOptimization, io)) {
    set_workspace(Workspace::Optimization);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceSystems, io)) {
    set_workspace(Workspace::Systems);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceCustom, io)) {
    set_workspace(Workspace::Custom);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceChat, io)) {
    set_workspace(Workspace::Chat);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceTraining, io)) {
    set_workspace(Workspace::Training);
  }
  if (shortcut_manager_.IsTriggered(ui::ActionId::WorkspaceContext, io)) {
    set_workspace(Workspace::Context);
  }
}

int App::Run() {
  if (!InitWindow()) return 1;
  if (!InitImGui()) return 1;
  RefreshData("initial");

  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    if (state_.should_refresh) {
      RefreshData("manual");
      state_.should_refresh = false;
    }
    
    MaybeAutoRefresh();
    RenderFrame();
  }
  return 0;
}

void App::RenderFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  shortcut_manager_.HandleCapture(ImGui::GetIO());
  HandleShortcuts();
  TickSimulatedMetrics(ImGui::GetIO().DeltaTime);

  glfwGetFramebufferSize(window_, &window_width_, &window_height_);

  auto refresh_cb = [this](const char* reason) { state_.should_refresh = true; };
  auto quit_cb = [this]() { glfwSetWindowShouldClose(window_, GLFW_TRUE); };

  ui::RenderMenuBar(state_, refresh_cb, quit_cb, shortcut_manager_,
                    &show_sample_review_, &show_shortcuts_window_);
  RenderLayout();

  if (show_sample_review_) {
    sample_review_.Render(&show_sample_review_);
  }

  ui::RenderShortcutsWindow(shortcut_manager_, &show_shortcuts_window_);
  shortcut_manager_.SaveIfDirty();

  if (state_.show_demo_window) {
    ImGui::ShowDemoWindow(&state_.show_demo_window);
    ImPlot::ShowDemoWindow();
  }

  ImGui::Render();
  glViewport(0, 0, window_width_, window_height_);
  glClearColor(0.07f, 0.07f, 0.09f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow* backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }
  glfwSwapBuffers(window_);
}

void App::RenderLayout() {
  bool docking_active = ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable;
  if (docking_active) {
    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

    if (state_.force_reset_layout || !ImGui::DockBuilderGetNode(dockspace_id)) {
      state_.force_reset_layout = false;
      ImGui::DockBuilderRemoveNode(dockspace_id);
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
      ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

      ImGuiID dock_main_id = dockspace_id;
      ImGuiID dock_right_id = dockspace_id;
      ImGuiID dock_bottom_id = dockspace_id;
      bool want_systems_panel = state_.show_systems_panel && state_.current_workspace == Workspace::Systems;

      if (state_.show_inspector || want_systems_panel) {
        dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.28f, nullptr, &dock_main_id);
      }
      if (state_.show_dataset_panel) {
        dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.30f, nullptr, &dock_main_id);
      }

      ImGui::DockBuilderDockWindow("WorkspaceContent", dock_main_id);
      if (state_.show_inspector) ImGui::DockBuilderDockWindow("InspectorPanel", dock_right_id);
      if (want_systems_panel) ImGui::DockBuilderDockWindow("SystemsPanel", dock_right_id);
      if (state_.show_dataset_panel) ImGui::DockBuilderDockWindow("DatasetPanel", dock_bottom_id);
      ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
    if (state_.lock_layout) dockspace_flags |= ImGuiDockNodeFlags_NoResize | ImGuiDockNodeFlags_NoSplit;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("MainDockSpaceHost", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_TitleBgActive]);
    ImGui::BeginChild("StaticSidebar", ImVec2(180, 0), true, ImGuiWindowFlags_NoScrollbar);
    ui::RenderSidebar(state_, font_ui_, font_header_);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::SameLine();

    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    ImGui::End(); // MainDockSpaceHost
  }

  // Common Panels
  if (state_.show_inspector) {
    ImGui::Begin("InspectorPanel", &state_.show_inspector);
    ui::RenderInspectorPanel(state_, loader_, font_header_, data_path_);
    ImGui::End();
  }
  if (state_.show_dataset_panel) {
    ImGui::Begin("DatasetPanel", &state_.show_dataset_panel);
    ui::RenderDatasetPanel(state_, loader_);
    ImGui::End();
  }
  if (state_.show_systems_panel && state_.current_workspace == Workspace::Systems) {
    ImGui::Begin("SystemsPanel", &state_.show_systems_panel);
    ui::RenderSystemsPanel(state_, font_header_, [this](const char* r){ RefreshData(r); });
    ImGui::End();
  }

  // Workspace Content
  ImGui::Begin("WorkspaceContent", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | (state_.allow_workspace_scroll ? 0 : ImGuiWindowFlags_NoScrollbar));
  switch (state_.current_workspace) {
    case Workspace::Dashboard: RenderDashboardView(); break;
    case Workspace::Analysis: RenderAnalysisView(); break;
    case Workspace::Optimization: RenderOptimizationView(); break;
    case Workspace::Systems: RenderSystemsView(); break;
    case Workspace::Custom: RenderCustomGridView(); break;
    case Workspace::Chat: RenderChatView(); break;
    case Workspace::Training: RenderTrainingView(); break;
    case Workspace::Context: RenderContextView(); break;
    case Workspace::Models: RenderModelsView(); break;
    default: break;
  }
  ImGui::End();

  if (state_.show_status_strip) {
    ImGui::SetNextWindowPos(ImVec2(0, (float)window_height_ - 24));
    ImGui::SetNextWindowSize(ImVec2((float)window_width_, 24));
    ImGui::Begin("StatusBar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDocking);
    ui::RenderStatusBar(state_, loader_, data_path_);
    ImGui::End();
  }
}

void App::RenderDashboardView() {
  ui::RenderSummaryRow(state_, loader_, font_ui_, font_header_);
  ImGui::Spacing();

  int columns = state_.auto_chart_columns ? (window_width_ > 1000 ? 3 : 2) : state_.chart_columns;
  if (ImGui::BeginTable("DashboardGrid", columns, ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableNextColumn(); ui::RenderQualityChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderGeneratorChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderCoverageChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderTrainingChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderAgentThroughputChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderLatentSpaceChart(state_, loader_);
    ImGui::EndTable();
  }
}

void App::RenderAnalysisView() {
  if (ImGui::BeginTable("AnalysisGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextColumn(); ui::RenderQualityChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderTrainingLossChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderGeneratorMixChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderEmbeddingQualityChart(state_, loader_);
    ImGui::EndTable();
  }
}

void App::RenderOptimizationView() {
  if (ImGui::BeginTable("OptimizationGrid", 2, ImGuiTableFlags_Resizable)) {
      ImGui::TableNextColumn(); ui::RenderEffectivenessChart(state_, loader_);
      ImGui::TableNextColumn(); ui::RenderThresholdOptimizationChart(state_, loader_);
      ImGui::TableNextColumn(); ui::RenderRejectionChart(state_, loader_);
      ImGui::TableNextColumn(); ui::RenderDomainCoverageChart(state_, loader_);
      ImGui::EndTable();
  }
}

void App::RenderSystemsView() {
  if (ImGui::BeginTable("SystemsGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextColumn(); ui::RenderAgentUtilizationChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderMissionProgressChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderMissionQueueChart(state_, loader_);
    ImGui::TableNextColumn(); ui::RenderAgentThroughputChart(state_, loader_);
    ImGui::EndTable();
  }
}

void App::RenderCustomGridView() {
  // Logic for custom grid rendering using RenderPlotByKind
}

void App::RenderChatView() {
  ImGui::BeginChild("ChatLayout", ImVec2(0, 0), true);
  ImGui::Columns(2, "ChatSplit", true);
  ImGui::SetColumnWidth(0, 300);
  ui::RenderKnobsTab(state_, loader_, data_path_, [this](const char* r){ RefreshData(r); }, nullptr);
  ImGui::NextColumn();
  ui::RenderLogsTab(state_, nullptr);
  ImGui::Columns(1);
  ImGui::EndChild();
}

void App::RenderTrainingView() {
  if (ImGui::BeginTabBar("TrainingTabs")) {
    if (ImGui::BeginTabItem("Dashboard")) { RenderDashboardView(); ImGui::EndTabItem(); }
    if (ImGui::BeginTabItem("Remote Training")) {
      training_dashboard_widget_.Render();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Agents")) { ui::RenderAgentsTab(state_, font_ui_, font_header_, nullptr); ImGui::EndTabItem(); }
    if (ImGui::BeginTabItem("Missions")) { ui::RenderMissionsTab(state_, nullptr); ImGui::EndTabItem(); }
    if (ImGui::BeginTabItem("Services")) { ui::RenderServicesTab(state_, nullptr); ImGui::EndTabItem(); }
    if (ImGui::BeginTabItem("Tables")) { ui::RenderTablesTab(state_, loader_); ImGui::EndTabItem(); }
    ImGui::EndTabBar();
  }
}

void App::RenderContextView() {
  ui::RenderContextTab(state_, text_editor_, memory_editor_, nullptr);
}

void App::RenderModelsView() {
  model_registry_widget_.Render();
}

void App::RenderExpandedPlot() {
    if (state_.expanded_plot == PlotKind::None) return;
    ImGui::OpenPopup("Expanded Plot");
    if (ImGui::BeginPopupModal("Expanded Plot", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ui::RenderPlotByKind(state_.expanded_plot, state_, loader_);
        if (ImGui::Button("Close")) state_.expanded_plot = PlotKind::None;
        ImGui::EndPopup();
    }
}

} // namespace viz
} // namespace hafs
