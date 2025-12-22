#include "app.h"
#include "core/logger.h"
#include "core/context.h"
#include "core/assets.h"

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

App::App(const std::string& data_path)
    : data_path_(data_path), loader_(data_path) {
  LOG_INFO("HAFS Studio initialize with data path: " + data_path);

  std::snprintf(state_.new_agent_role.data(), state_.new_agent_role.size(), "Evaluator");
  std::snprintf(state_.new_mission_owner.data(), state_.new_mission_owner.size(), "Ops");
  std::snprintf(state_.system_prompt.data(), state_.system_prompt.size(), 
                "You are a HAFS data science assistant. Analyze the training trends and suggest optimizations.");
  
  const char* home = std::getenv("HOME");
  state_.current_browser_path = home ? std::filesystem::path(home) : std::filesystem::current_path();
  ui::RefreshBrowserEntries(state_);
  SeedDefaultState();

  // Create graphics context
  context_ = std::make_unique<studio::core::GraphicsContext>("HAFS Studio", 1400, 900);
  if (context_->IsValid()) {
      fonts_ = studio::core::AssetLoader::LoadFonts();
      themes::ApplyHafsTheme();
      shortcut_manager_.LoadFromDisk();

      // Better default heights for richness
      state_.chart_height = 220.0f;
      state_.plot_height = 220.0f;
      state_.chart_columns = 2;
  } else {
      LOG_ERROR("Failed to initialize graphics context");
  }
}

int App::Run() {
  if (!context_ || !context_->IsValid()) return 1;

  RefreshData("startup");

  double last_time = glfwGetTime();
  while (!context_->ShouldClose()) {
    context_->PollEvents();

    double current_time = glfwGetTime();
    float dt = static_cast<float>(current_time - last_time);
    last_time = current_time;

    TickSimulatedMetrics(dt);

    if (state_.auto_refresh && (current_time - state_.last_refresh_time > state_.refresh_interval_sec)) {
      RefreshData("auto");
    }

    if (state_.should_refresh) {
      RefreshData("manual");
      state_.should_refresh = false;
    }

    RenderFrame();
  }

  return 0;
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
  } else if (!status.AnyOk() && !status.FoundCount()) {
    msg = "No data sources found (" + std::string(reason) + ")";
  } else if (!ok) {
    msg = "Data refresh failed (" + std::string(reason) + ")";
  } else {
    msg = "Data refreshed (" + std::string(reason) + ")";
  }
  ui::AppendLog(state_, "system", msg, "system");
  LOG_INFO(msg);
}

void App::SyncDataBackedState() {
  const auto& coverage = loader_.GetCoverage();
  const auto& trends = loader_.GetQualityTrends();
  const auto& runs = loader_.GetTrainingRuns();
  const auto& generators = loader_.GetGeneratorStats();

  // Sync Agents
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

  float avg_loss = 0.0f;
  for (const auto& run : runs) avg_loss += run.final_loss;
  if (!runs.empty()) avg_loss /= static_cast<float>(runs.size());

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
  trainer->success_rate = avg_loss > 0.0f ? ui::Clamp01(1.0f / (1.0f + avg_loss)) : 0.0f;
  trainer->status = "Active";

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
}

void App::SeedDefaultState() {
  ui::AppendLog(state_, "system", "HAFS Studio environment ready.", "system");
  state_.sparkline_data.resize(30, 0.0f);
  for (float& f : state_.sparkline_data) f = (float)(rand() % 100) / 100.0f;
}

void App::TickSimulatedMetrics(float dt) {
  state_.pulse_timer += dt;
  if (!state_.simulate_activity) return;

  for (auto& agent : state_.agents) {
    if (agent.data_backed || !agent.enabled) continue;
    agent.activity_phase += dt * (0.5f + (float)(rand() % 100) / 100.0f);
    agent.cpu_pct = 20.0f + 15.0f * (1.0f + sinf(agent.activity_phase));
    agent.mem_pct = 15.0f + 5.0f * (1.0f + cosf(agent.activity_phase * 0.7f));
  }
}

void App::RenderFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  const ImGuiIO& io = ImGui::GetIO();
  if (shortcut_manager_.IsTriggered(ui::ActionId::Refresh, io)) state_.should_refresh = true;

  // New: Layout Presets
  if (ImGui::IsKeyPressed(ImGuiKey_F1)) { state_.layout_preset = 0; state_.force_reset_layout = true; }
  if (ImGui::IsKeyPressed(ImGuiKey_F2)) { state_.layout_preset = 1; state_.force_reset_layout = true; }
  if (ImGui::IsKeyPressed(ImGuiKey_F3)) { state_.layout_preset = 2; state_.force_reset_layout = true; }

  auto refresh_cb = [this](const char* reason) { state_.should_refresh = true; };
  auto quit_cb = [this]() { glfwSetWindowShouldClose(context_->GetWindow(), true); };

  ui::RenderMenuBar(state_, refresh_cb, quit_cb, shortcut_manager_, &show_sample_review_, &show_shortcuts_window_);
  RenderLayout();

  if (show_sample_review_) sample_review_.Render(&show_sample_review_);
  ui::RenderShortcutsWindow(shortcut_manager_, &show_shortcuts_window_);
  shortcut_manager_.SaveIfDirty();

  RenderExpandedPlot();
  RenderFloaters();

  if (state_.show_demo_window) {
    ImGui::ShowDemoWindow(&state_.show_demo_window);
    ImPlot::ShowDemoWindow();
  }

  // Finalize ImGui Frame
  ImGui::Render();
  int w, h;
  glfwGetFramebufferSize(context_->GetWindow(), &w, &h);
  glViewport(0, 0, w, h);
  glClearColor(0.07f, 0.07f, 0.09f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow* backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }
  context_->SwapBuffers();
}

void App::RenderLayout() {
  bool docking_active = ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable;
  if (docking_active) {
    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

    if (state_.force_reset_layout || !ImGui::DockBuilderGetNode(dockspace_id)) {
      state_.force_reset_layout = false;
      ImGui::DockBuilderRemoveNode(dockspace_id);
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
      
      ImGuiID dock_main_id = dockspace_id;
      ImGuiID dock_right_id = 0;
      ImGuiID dock_bottom_id = 0;
      ImGuiID dock_left_id = 0;
      
      if (state_.layout_preset == 1) { // Analyst Layout
          dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.40f, nullptr, &dock_main_id);
          dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.20f, nullptr, &dock_main_id);
      } else if (state_.layout_preset == 2) { // System Layout
          dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.50f, nullptr, &dock_main_id);
      } else { // Default
          dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.25f, nullptr, &dock_main_id);
          dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.35f, nullptr, &dock_main_id);
          dock_left_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.18f, nullptr, &dock_main_id);
          ImGui::DockBuilderDockWindow("StaticSidebar", dock_left_id);
      }

      ImGui::DockBuilderDockWindow("WorkspaceContent", dock_main_id);
      ImGui::DockBuilderDockWindow("InspectorPanel", dock_right_id);
      ImGui::DockBuilderDockWindow("SystemsPanel", dock_right_id);
      ImGui::DockBuilderDockWindow("DatasetPanel", dock_bottom_id);
      ImGui::DockBuilderFinish(dockspace_id);
    }

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
    ImGui::Begin("StaticSidebar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
    ui::RenderSidebar(state_, fonts_.ui, fonts_.header);
    ImGui::End();
    ImGui::PopStyleColor();

    float status_bar_height = state_.show_status_strip ? 28.0f : 0.0f;
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, -status_bar_height), ImGuiDockNodeFlags_None);

    if (state_.show_status_strip) {
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - status_bar_height);
        ImGui::Separator();
        ImGui::BeginChild("StatusBarHost", ImVec2(0, status_bar_height), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoInputs);
        ui::RenderStatusBar(state_, loader_, data_path_);
        ImGui::EndChild();
    }

    ImGui::End();
  }

  if (state_.show_inspector) {
    ImGui::Begin("InspectorPanel", &state_.show_inspector);
    ui::RenderInspectorPanel(state_, loader_, fonts_.header, data_path_);
    ImGui::End();
  }
  if (state_.show_dataset_panel) {
    ImGui::Begin("DatasetPanel", &state_.show_dataset_panel);
    ui::RenderDatasetPanel(state_, loader_);
    ImGui::End();
  }
  if (state_.show_systems_panel && state_.current_workspace == Workspace::Systems) {
    ImGui::Begin("SystemsPanel", &state_.show_systems_panel);
    ui::RenderSystemsPanel(state_, fonts_.header, [this](const char* r){ RefreshData(r); });
    ImGui::End();
  }

  ImGui::Begin("WorkspaceContent", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
  switch (state_.current_workspace) {
    case Workspace::Dashboard: RenderDashboardView(); break;
    case Workspace::Analysis: RenderAnalysisView(); break;
    case Workspace::Optimization: RenderOptimizationView(); break;
    case Workspace::Systems: RenderSystemsView(); break;
    case Workspace::Chat: RenderChatView(); break;
    case Workspace::Training: RenderTrainingView(); break;
    case Workspace::Context: RenderContextView(); break;
    case Workspace::Models: RenderModelsView(); break;
    default: break;
  }
  ImGui::End();

}

void App::RenderDashboardView() {
  ui::RenderSummaryRow(state_, loader_, fonts_.ui, fonts_.header);
  ImGui::Spacing();

  int columns = state_.chart_columns;
  if (ImGui::BeginTable("DashboardGrid", columns, ImGuiTableFlags_Resizable)) {
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
    ui::RenderComparisonView(state_, loader_, fonts_.ui, fonts_.header);
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
    if (ImGui::BeginTabItem("Remote Training")) { training_dashboard_widget_.Render(); ImGui::EndTabItem(); }
    if (ImGui::BeginTabItem("Agents")) { ui::RenderAgentsTab(state_, fonts_.ui, fonts_.header, nullptr); ImGui::EndTabItem(); }
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
void App::RenderFloaters() {
    auto it = state_.active_floaters.begin();
    while (it != state_.active_floaters.end()) {
        PlotKind kind = *it;
        std::string title = std::string("Floater##") + std::to_string(static_cast<int>(kind));
        
        bool open = true;
        ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ui::RenderPlotByKind(kind, state_, loader_);
        }
        ImGui::End();

        if (!open) {
            it = state_.active_floaters.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace viz
} // namespace hafs
