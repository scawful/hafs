#include "app.h"

#include <algorithm>
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

namespace hafs {
namespace viz {

namespace {

void GlfwErrorCallback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

AgentState* FindAgentByName(std::vector<AgentState>& agents,
                            const std::string& name) {
  for (auto& agent : agents) {
    if (agent.name == name) return &agent;
  }
  return nullptr;
}

float Clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

}  // namespace

App::App(const std::string& data_path)
    : data_path_(data_path), loader_(data_path) {
  std::snprintf(new_agent_role_.data(), new_agent_role_.size(), "Evaluator");
  std::snprintf(new_mission_owner_.data(), new_mission_owner_.size(), "Ops");
  std::snprintf(system_prompt_.data(), system_prompt_.size(), 
                "You are a HAFS data science assistant. Analyze the training trends and suggest optimizations.");
  
  const char* home = std::getenv("HOME");
  current_browser_path_ = home ? std::filesystem::path(home) : std::filesystem::current_path();
  RefreshBrowserEntries();
  SeedDefaultState();
}

App::~App() { Shutdown(); }

bool App::InitWindow() {
  glfwSetErrorCallback(GlfwErrorCallback);
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return false;
  }

  // OpenGL 3.2 Core Profile for macOS compatibility
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  window_ = glfwCreateWindow(window_width_, window_height_,
                             "HAFS Training Data Visualization", nullptr, nullptr);
  if (!window_) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // Enable vsync

  return true;
}

bool App::InitImGui() {
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  imgui_ctx_ = ImGui::CreateContext();
  implot_ctx_ = ImPlot::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Typography - Load Roboto if available in build deps
  std::vector<std::string> search_paths = {
      "../../_deps/imgui-src/misc/fonts/Roboto-Medium.ttf",
      "_deps/imgui-src/misc/fonts/Roboto-Medium.ttf",
      "../_deps/imgui-src/misc/fonts/Roboto-Medium.ttf",
      "../../../_deps/imgui-src/misc/fonts/Roboto-Medium.ttf"
  };

  const char* font_path = nullptr;
  for (const auto& path : search_paths) {
      if (std::filesystem::exists(path)) {
          font_path = path.c_str();
          break;
      }
  }

  if (font_path) {
      // 1. Load UI Font (16px)
      font_ui_ = io.Fonts->AddFontFromFileTTF(font_path, 16.0f);
      
      // 2. Load Icons and Merge into UI Font
      std::vector<std::string> icon_search_paths = {
          "../../../../src/cc/viz/assets/font/MaterialIcons-Regular.ttf",
          "../../../src/cc/viz/assets/font/MaterialIcons-Regular.ttf",
          "../../src/cc/viz/assets/font/MaterialIcons-Regular.ttf",
          "assets/font/MaterialIcons-Regular.ttf",
          "../assets/font/MaterialIcons-Regular.ttf"
      };

      const char* icon_font_path = nullptr;
      for (const auto& path : icon_search_paths) {
          if (std::filesystem::exists(path)) {
              icon_font_path = path.c_str();
              break;
          }
      }

      if (icon_font_path) {
          ImFontConfig icons_config;
          icons_config.MergeMode = true;
          icons_config.PixelSnapH = true;
          icons_config.GlyphMinAdvanceX = 13.0f;
          icons_config.GlyphOffset = ImVec2(0, 5.0f);
          static const ImWchar icon_ranges[] = { ICON_MIN_MD, 0xf900, 0 };
          
          // Match yaze: Load icons after each primary font
          io.Fonts->AddFontFromFileTTF(icon_font_path, 18.0f, &icons_config, icon_ranges);
          
          font_header_ = io.Fonts->AddFontFromFileTTF(font_path, 20.0f);
          icons_config.GlyphOffset = ImVec2(0, 6.0f);
          io.Fonts->AddFontFromFileTTF(icon_font_path, 20.0f, &icons_config, icon_ranges);

          printf("Loaded Icons: Material Design (Synced with yaze params from %s)\n", icon_font_path);
      } else {
          printf("Warning: Material Icons font not found in search paths.\n");
          font_header_ = io.Fonts->AddFontFromFileTTF(font_path, 20.0f);
      }
      
      printf("Loaded Typography: Roboto (Medium from %s)\n", font_path);
  } else {
      io.Fonts->AddFontDefault();
      printf("Warning: Roboto font not found. Using default font.\n");
  }

  // Setup style - Luxe profiles
  themes::ApplyHafsTheme(current_theme_);

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 150");

  return true;
}

void App::Shutdown() {
  if (imgui_ctx_) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext(implot_ctx_);
    ImGui::DestroyContext(imgui_ctx_);
    imgui_ctx_ = nullptr;
    implot_ctx_ = nullptr;
  }

  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
  }
}

void App::RefreshData(const char* reason) {
  loader_.Refresh();
  SyncDataBackedState();
  last_refresh_time_ = glfwGetTime();

  // Initialize Knowledge Graph if empty
  if (knowledge_concepts_.empty()) {
    knowledge_concepts_ = {
      "Movement", "Combat", "Inventory", "NPC", "Dialogue", 
      "Pathfinding", "Logic", "Memory", "Animation"
    };
    for (size_t i = 0; i < knowledge_concepts_.size(); ++i) {
      knowledge_nodes_x_.push_back(10.0f + 80.0f * (float)rand() / RAND_MAX);
      knowledge_nodes_y_.push_back(10.0f + 80.0f * (float)rand() / RAND_MAX);
    }
    for (size_t i = 0; i < knowledge_concepts_.size(); ++i) {
      for (size_t j = i + 1; j < knowledge_concepts_.size(); ++j) {
        if ((float)rand() / RAND_MAX > 0.85f) {
           knowledge_edges_.push_back({(int)i, (int)j});
        }
      }
    }
  }

  if (!reason) return;
  if (std::strcmp(reason, "auto") == 0 && !verbose_logs_) return;

  std::string note = "Data refresh";
  if (std::strlen(reason) > 0) {
    note += " (";
    note += reason;
    note += ")";
  }
  AppendLog("system", note, "system");
}

void App::MaybeAutoRefresh() {
  if (!auto_refresh_) return;
  double now = glfwGetTime();
  if (now - last_refresh_time_ >= refresh_interval_sec_) {
    RefreshData("auto");
  }
}

void App::SeedDefaultState() {
  if (!agents_.empty() || !missions_.empty()) return;

  agents_.push_back(AgentState{"Coordinator",
                               "Planner",
                               "Active",
                               true,
                               false,
                               2,
                               128,
                               0.94f,
                               24.0f,
                               18.0f,
                               22.0f,
                               0.2f});
  agents_.push_back(AgentState{"Embedding Indexer",
                               "Indexer",
                               "Idle",
                               true,
                               false,
                               4,
                               420,
                               0.88f,
                               18.0f,
                               32.0f,
                               28.0f,
                               0.8f});
  agents_.push_back(AgentState{"Quality Monitor",
                               "Evaluator",
                               "Idle",
                               true,
                               false,
                               1,
                               96,
                               0.91f,
                               22.0f,
                               26.0f,
                               18.0f,
                               1.6f});
  agents_.push_back(AgentState{"Trainer Coordinator",
                               "Trainer",
                               "Idle",
                               true,
                               false,
                               3,
                               64,
                               0.86f,
                               30.0f,
                               34.0f,
                               36.0f,
                               2.3f});

  missions_.push_back(
      MissionState{"Bootstrap Embeddings", "Indexer", "Active", false, 4, 0.35f});
  missions_.push_back(
      MissionState{"Audit Quality Loop", "Evaluator", "Queued", false, 3, 0.05f});
  missions_.push_back(
      MissionState{"Stabilize Trainer", "Trainer", "Active", false, 5, 0.55f});

  AppendLog("system", "Ops console initialized.", "system");
  AppendLog("Coordinator", "Background agents online.", "agent");
}

void App::SyncDataBackedState() {
  SeedDefaultState();

  const auto& generator_stats = loader_.GetGeneratorStats();
  for (const auto& stats : generator_stats) {
    AgentState* agent = FindAgentByName(agents_, stats.name);
    if (!agent) {
      agents_.push_back(AgentState{});
      agent = &agents_.back();
      agent->name = stats.name;
      agent->role = "Generator";
      agent->activity_phase = 0.4f * static_cast<float>(agents_.size());
    }

    agent->data_backed = true;
    agent->enabled = true;
    agent->tasks_completed = stats.samples_generated;
    agent->queue_depth = stats.samples_rejected;
    agent->success_rate = Clamp01(stats.acceptance_rate);
    agent->avg_latency_ms = 12.0f + (1.0f - agent->success_rate) * 85.0f;
    agent->cpu_pct =
        std::min(90.0f, 20.0f + std::fmod(stats.samples_generated * 0.7f, 70.0f));
    agent->mem_pct =
        std::min(90.0f, 15.0f + std::fmod(stats.samples_accepted * 0.5f, 70.0f));
    agent->status = agent->queue_depth > 0 ? "Busy" : "Idle";
  }

  const auto& coverage = loader_.GetCoverage();
  AgentState* indexer = FindAgentByName(agents_, "Embedding Indexer");
  if (!indexer) {
    agents_.push_back(AgentState{});
    indexer = &agents_.back();
    indexer->name = "Embedding Indexer";
    indexer->role = "Indexer";
    indexer->activity_phase = 0.8f;
  }
  indexer->data_backed = true;
  indexer->enabled = true;
  indexer->tasks_completed = coverage.total_samples;
  indexer->queue_depth = coverage.sparse_regions;
  indexer->success_rate = Clamp01(coverage.coverage_score);
  indexer->avg_latency_ms = 18.0f + (1.0f - indexer->success_rate) * 55.0f;
  indexer->cpu_pct =
      std::min(92.0f, 25.0f + static_cast<float>(coverage.num_regions) * 0.4f);
  indexer->mem_pct = std::min(
      92.0f, 20.0f + static_cast<float>(coverage.total_samples) * 0.001f);
  indexer->status = indexer->queue_depth > 0 ? "Busy" : "Idle";

  const auto& trends = loader_.GetQualityTrends();
  float quality_mean = 0.0f;
  int insufficient = 0;
  for (const auto& trend : trends) {
    quality_mean += trend.mean;
    if (trend.trend_direction == "insufficient") ++insufficient;
  }
  if (!trends.empty()) quality_mean /= static_cast<float>(trends.size());

  AgentState* evaluator = FindAgentByName(agents_, "Quality Monitor");
  if (!evaluator) {
    agents_.push_back(AgentState{});
    evaluator = &agents_.back();
    evaluator->name = "Quality Monitor";
    evaluator->role = "Evaluator";
    evaluator->activity_phase = 1.4f;
  }
  evaluator->data_backed = true;
  evaluator->enabled = true;
  evaluator->tasks_completed = static_cast<int>(trends.size());
  evaluator->queue_depth = insufficient;
  evaluator->success_rate = Clamp01(quality_mean);
  evaluator->avg_latency_ms = 20.0f + (1.0f - evaluator->success_rate) * 70.0f;
  evaluator->cpu_pct =
      std::min(88.0f, 20.0f + static_cast<float>(trends.size()) * 2.0f);
  evaluator->mem_pct = std::min(88.0f, 15.0f + insufficient * 6.0f);
  evaluator->status = evaluator->queue_depth > 0 ? "Review" : "Idle";

  const auto& runs = loader_.GetTrainingRuns();
  float avg_loss = 0.0f;
  for (const auto& run : runs) avg_loss += run.final_loss;
  if (!runs.empty()) avg_loss /= static_cast<float>(runs.size());

  int high_loss = 0;
  for (const auto& run : runs) {
    if (run.final_loss > avg_loss) ++high_loss;
  }

  AgentState* trainer = FindAgentByName(agents_, "Trainer Coordinator");
  if (!trainer) {
    agents_.push_back(AgentState{});
    trainer = &agents_.back();
    trainer->name = "Trainer Coordinator";
    trainer->role = "Trainer";
    trainer->activity_phase = 2.1f;
  }
  trainer->data_backed = true;
  trainer->enabled = true;
  trainer->tasks_completed = static_cast<int>(runs.size());
  trainer->queue_depth = high_loss;
  trainer->success_rate =
      avg_loss > 0.0f ? Clamp01(1.0f / (1.0f + avg_loss)) : 0.0f;
  trainer->avg_latency_ms = 28.0f + avg_loss * 60.0f;
  trainer->cpu_pct =
      std::min(90.0f, 30.0f + static_cast<float>(runs.size()) * 6.0f);
  trainer->mem_pct =
      std::min(90.0f, 25.0f + static_cast<float>(runs.size()) * 4.0f);
  trainer->status = trainer->queue_depth > 0 ? "Training" : "Idle";

  missions_.erase(
      std::remove_if(missions_.begin(), missions_.end(),
                     [](const MissionState& mission) {
                       return mission.data_backed;
                     }),
      missions_.end());

  for (const auto& run : runs) {
    MissionState mission;
    mission.data_backed = true;
    mission.owner = run.model_name.empty() ? "Trainer" : run.model_name;
    mission.name =
        run.run_id.size() > 12 ? run.run_id.substr(0, 12) : run.run_id;
    mission.status = "Complete";
    mission.priority = run.final_loss > avg_loss ? 4 : 3;
    mission.progress = 1.0f;
    missions_.push_back(std::move(mission));
  }
}

void App::TickSimulatedMetrics(float dt) {
  pulse_timer_ += dt;
  if (pulse_timer_ > 6.28318f) pulse_timer_ -= 6.28318f;

  if (!simulate_activity_) return;

  float time = static_cast<float>(glfwGetTime());
  for (auto& agent : agents_) {
    if (agent.data_backed || !agent.enabled) continue;
    float wave = 0.5f + 0.5f * std::sin(time + agent.activity_phase);
    int queue_delta =
        static_cast<int>(wave * 3.0f * agent_activity_scale_) - 1;
    agent.queue_depth = std::max(0, agent.queue_depth + queue_delta);
    agent.tasks_completed +=
        static_cast<int>(wave * 2.0f * agent_activity_scale_);
    agent.success_rate = Clamp01(agent.success_rate + (wave - 0.5f) * 0.02f);
    agent.avg_latency_ms = std::max(
        6.0f, agent.avg_latency_ms + (0.5f - wave) * 2.0f);
    agent.cpu_pct =
        std::min(95.0f, std::max(5.0f, agent.cpu_pct + (wave - 0.5f) * 6.0f));
    agent.mem_pct =
        std::min(95.0f, std::max(5.0f, agent.mem_pct + (wave - 0.5f) * 4.0f));
    agent.status = agent.queue_depth > 0 ? "Busy" : "Idle";
  }

  for (auto& mission : missions_) {
    if (mission.data_backed) continue;
    if (mission.status == "Complete") continue;

    float rate =
        (0.08f + 0.04f * static_cast<float>(mission.priority)) *
        mission_priority_bias_;
    mission.progress = Clamp01(mission.progress + dt * rate);
    if (mission.progress > 0.02f && mission.status == "Queued") {
      mission.status = "Active";
    }
    if (mission.progress >= 1.0f) {
      mission.status = "Complete";
      AppendLog("system", "Mission complete: " + mission.name, "system");
    }
  }
}

void App::AppendLog(const std::string& agent, const std::string& message,
                    const std::string& kind) {
  if (message.empty()) return;
  constexpr size_t kMaxLogs = 300;
  logs_.push_back(LogEntry{agent, message, kind});
  if (logs_.size() > kMaxLogs) logs_.pop_front();
}

int App::Run() {
  if (!InitWindow()) return 1;
  if (!InitImGui()) return 1;

  // Initial data load
  RefreshData("initial");

  // Main loop
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    // Handle refresh request
    if (should_refresh_) {
      RefreshData("manual");
      should_refresh_ = false;
    }

    // Handle F5 for refresh
    if (glfwGetKey(window_, GLFW_KEY_F5) == GLFW_PRESS) {
      should_refresh_ = true;
    }

    MaybeAutoRefresh();
    RenderFrame();
  }

  return 0;
}

void App::RenderFrame() {
  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  TickSimulatedMetrics(ImGui::GetIO().DeltaTime);

  // Get window size for full-screen layout
  glfwGetFramebufferSize(window_, &window_width_, &window_height_);

  RenderMenuBar();
  RenderLayout();

  // Demo windows for development
  if (show_demo_window_) {
    ImGui::ShowDemoWindow(&show_demo_window_);
    ImPlot::ShowDemoWindow();
  }

  // Rendering
  ImGui::Render();
  glViewport(0, 0, window_width_, window_height_);
  glClearColor(0.07f, 0.07f, 0.09f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  // Update and Render additional Platform Windows
  if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow* backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }

  glfwSwapBuffers(window_);
}

void App::RenderMenuBar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Refresh", "F5")) {
        should_refresh_ = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Quit", "Ctrl+Q")) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::BeginMenu("Workspace")) {
        if (ImGui::MenuItem("Dashboard", "1", current_workspace_ == Workspace::Dashboard)) {
          current_workspace_ = Workspace::Dashboard;
          force_reset_layout_ = true;
        }
        if (ImGui::MenuItem("Analysis", "2", current_workspace_ == Workspace::Analysis)) {
          current_workspace_ = Workspace::Analysis;
          force_reset_layout_ = true;
        }
        if (ImGui::MenuItem("Optimization", "3", current_workspace_ == Workspace::Optimization)) {
          current_workspace_ = Workspace::Optimization;
          force_reset_layout_ = true;
        }
        if (ImGui::MenuItem("Systems", "4", current_workspace_ == Workspace::Systems)) {
          current_workspace_ = Workspace::Systems;
          force_reset_layout_ = true;
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Theme Profile")) {
        if (ImGui::MenuItem("Cobalt", nullptr, current_theme_ == ThemeProfile::Cobalt)) {
            current_theme_ = ThemeProfile::Cobalt;
            themes::ApplyHafsTheme(current_theme_);
        }
        if (ImGui::MenuItem("Amber", nullptr, current_theme_ == ThemeProfile::Amber)) {
            current_theme_ = ThemeProfile::Amber;
            themes::ApplyHafsTheme(current_theme_);
        }
        if (ImGui::MenuItem("Emerald", nullptr, current_theme_ == ThemeProfile::Emerald)) {
            current_theme_ = ThemeProfile::Emerald;
            themes::ApplyHafsTheme(current_theme_);
        }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Lock Layout", nullptr, lock_layout_)) lock_layout_ = !lock_layout_;
      if (ImGui::MenuItem("Reset Layout")) force_reset_layout_ = true;
      ImGui::Separator();
      ImGui::MenuItem("Show Demo Windows", nullptr, &show_demo_window_);
      ImGui::MenuItem("Compact Charts", nullptr, &compact_charts_);
      ImGui::MenuItem("Show Status Strip", nullptr, &show_status_strip_);
      ImGui::MenuItem("Show Controls", nullptr, &show_controls_);
      ImGui::MenuItem("Simulate Activity", nullptr, &simulate_activity_);
      ImGui::MenuItem("Auto Refresh", nullptr, &auto_refresh_);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("About")) {
        // TODO: Show about dialog
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }
}

void App::RenderDockSpace() {
  static bool opt_fullscreen = true;
  static bool opt_padding = false;
  static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

  ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
  if (opt_fullscreen) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  } else {
    dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
  }

  if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
    window_flags |= ImGuiWindowFlags_NoBackground;

  if (!opt_padding) ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("HAFSDockSpace", nullptr, window_flags);
  if (!opt_padding) ImGui::PopStyleVar();
  if (opt_fullscreen) ImGui::PopStyleVar(2);

  ImGuiIO& io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
    
    if (force_reset_layout_ || !ImGui::DockBuilderGetNode(dockspace_id)) {
      force_reset_layout_ = false;
      ImGui::DockBuilderRemoveNode(dockspace_id);
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
      ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

      // Simplify docking: Only one main node for the active workspace
      ImGuiID dock_main_id = dockspace_id;
      ImGui::DockBuilderDockWindow("WorkspaceContent", dock_main_id);
      ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
    if (lock_layout_) dockspace_flags |= ImGuiDockNodeFlags_NoResize | ImGuiDockNodeFlags_NoSplit;
    
    // Render Static Sidebar inside the root window but outside the DockSpace
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_TitleBgActive]);
    ImGui::BeginChild("StaticSidebar", ImVec2(180, 0), true, ImGuiWindowFlags_NoScrollbar);
    RenderSidebar();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
  }

  ImGui::End();
}

void App::RenderLayout() {
  RenderDockSpace();
  
  // Enable scrolling for the workspace content
  ImGui::Begin("WorkspaceContent", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysVerticalScrollbar);
  
  // Header with Metrics
  RenderMetricCards();
  ImGui::Separator();
  ImGui::Spacing();

  // Active Workspace Grid
  switch (current_workspace_) {
    case Workspace::Dashboard:    RenderDashboardView(); break;
    case Workspace::Analysis:     RenderAnalysisView(); break;
    case Workspace::Optimization: RenderOptimizationView(); break;
    case Workspace::Systems:      RenderSystemsView(); break;
    case Workspace::Chat:         RenderChatView(); break;
    case Workspace::Training:     RenderTrainingView(); break;
    case Workspace::Context:      RenderContextView(); break;
  }

  ImGui::End();

  if (show_status_strip_) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + viewport->Size.y - 32));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 32));
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("StatusBar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);
    RenderStatusBar();
    ImGui::End();
    ImGui::PopStyleVar(2);
  }
}

void App::RenderSummaryRow() {
  RenderMetricCards();
  
  ImGui::Spacing();
  
  // Swarm Topology Overview
  if (ImGui::BeginChild("SwarmTopology", ImVec2(0, 120), true)) {
      if (font_header_) ImGui::PushFont(font_header_);
      ImGui::Text(ICON_MD_HUB " SWARM TOPOLOGY");
      if (font_header_) ImGui::PopFont();
      
      ImGui::Separator();
      
      if (ImGui::BeginTable("Topology", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
          ImGui::TableSetupColumn("Active Agents", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Queue Depth", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Mission Velocity", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Avg. Success", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Health", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableHeadersRow();
          
          int active_count = 0;
          float total_success = 0.0f;
          int total_queue = 0;
          for (const auto& a : agents_) {
              if (a.enabled) active_count++;
              total_success += a.success_rate;
              total_queue += a.queue_depth;
          }
          if (!agents_.empty()) total_success /= (float)agents_.size();
          
          ImGui::TableNextRow();
          
          // Column 0: Active Agents
          ImGui::TableSetColumnIndex(0);
          ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "%d / %d", active_count, (int)agents_.size());
          
          // Column 1: Queue Depth
          ImGui::TableSetColumnIndex(1);
          ImGui::Text("%d Tasks", total_queue);
          
          // Column 2: Mission Velocity
          ImGui::TableSetColumnIndex(2);
          float avg_progress = 0.0f;
          for (const auto& m : missions_) avg_progress += m.progress;
          if (!missions_.empty()) avg_progress /= (float)missions_.size();
          ImGui::ProgressBar(avg_progress, ImVec2(-1, 0), "");
          
          // Column 3: Success Rate
          ImGui::TableSetColumnIndex(3);
          ImGui::Text("%.1f%%", total_success * 100.0f);
          
          // Column 4: Health
          ImGui::TableSetColumnIndex(4);
          if (total_success > 0.9f) ImGui::TextColored(ImVec4(0, 1, 0, 1), ICON_MD_CHECK_CIRCLE " NOMINAL");
          else ImGui::TextColored(ImVec4(1, 1, 0, 1), ICON_MD_WARNING " CAUTION");
          
          ImGui::EndTable();
      }
      ImGui::EndChild();
  }
}

void App::RenderControlColumn() {
  ImGui::BeginChild("ControlColumn", ImVec2(0, 0), true);
  if (ImGui::BeginTabBar("ControlTabs")) {
    if (ImGui::BeginTabItem("Knobs")) {
      RenderKnobsTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Agents")) {
      RenderAgentsTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Services")) {
      RenderServicesTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Missions")) {
      RenderMissionsTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Context")) {
      RenderContextTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Prompts")) {
      RenderAgentPromptTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Tables")) {
      RenderTablesTab();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Logs")) {
      RenderLogsTab();
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::EndChild();
}

void App::RenderMetricsColumn() {
  ImGui::EndChild();
  ImGui::PopStyleVar(2);
}

void App::RenderKnobsTab() {
  ImGui::Text("Runtime Controls");
  if (ImGui::Button("Refresh Now")) {
    RefreshData("ui");
  }
  ImGui::SameLine();
  ImGui::Checkbox("Auto Refresh", &auto_refresh_);
  ImGui::SliderFloat("Refresh Interval (s)", &refresh_interval_sec_, 2.0f, 30.0f);

  ImGui::Separator();
  ImGui::Text("Simulation");
  ImGui::Checkbox("Simulate Activity", &simulate_activity_);
  ImGui::SliderFloat("Agent Activity", &agent_activity_scale_, 0.3f, 3.0f);
  ImGui::SliderFloat("Mission Bias", &mission_priority_bias_, 0.5f, 2.0f);

  ImGui::Separator();
  ImGui::Text("Visualization");
  ImGui::SliderFloat("Chart Height (compact)", &chart_height_, 130.0f, 240.0f);
  ImGui::Checkbox("Compact Charts", &compact_charts_);
  ImGui::Checkbox("Show Status Strip", &show_status_strip_);
  ImGui::Checkbox("Show Controls", &show_controls_);
  ImGui::Checkbox("Auto Columns", &auto_chart_columns_);
  if (!auto_chart_columns_) {
    ImGui::SliderInt("Chart Columns", &chart_columns_, 2, 4);
  } else {
    ImGui::TextDisabled("Chart Columns: auto");
  }
  ImGui::SliderFloat("Embedding Sample Rate", &embedding_sample_rate_, 0.1f, 1.0f);
  ImGui::SliderFloat("Quality Threshold", &quality_threshold_, 0.4f, 0.95f);
  ImGui::SliderInt("Mission Concurrency", &mission_concurrency_, 1, 12);
  ImGui::Checkbox("Verbose Logs", &verbose_logs_);
  ImGui::Checkbox("Pulse Animations", &use_pulse_animations_);
  ImGui::Checkbox("Data Scientist Mode", &data_scientist_mode_);
  ImGui::Checkbox("Show All Chart Windows", &show_all_charts_);
  if (data_scientist_mode_) {
      ImGui::Indent();
      ImGui::TextDisabled("Extending analytics and visuals...");
      ImGui::Unindent();
  }

  ImGui::Separator();
  ImGui::Text("ImPlot Features");
  static bool show_crosshairs = true;
  if (ImGui::Checkbox("Show Crosshairs", &show_crosshairs)) {
      // Logic for crosshairs if handled globally
  }
  static bool show_tooltips = true;
  ImGui::Checkbox("Enable Tooltips", &show_tooltips);
  
  ImGui::Separator();
  ImGui::Text("Advanced");
  if (ImGui::Button("Purge Mission Queue")) {
      missions_.clear();
      AppendLog("system", "Mission queue purged.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Force Reconnect")) {
      AppendLog("system", "Forcing backend reconnect...", "system");
  }

  ImGui::Separator();
  ImGui::TextDisabled("Data Path");
  ImGui::TextWrapped("%s", data_path_.c_str());
}

void App::RenderAgentsTab() {
  ImGui::Text("Background Agents");
  ImGui::InputTextWithHint("##AgentName", "Agent name",
                           new_agent_name_.data(), new_agent_name_.size());
  ImGui::InputTextWithHint("##AgentRole", "Role",
                           new_agent_role_.data(), new_agent_role_.size());

  if (ImGui::Button("Spawn Agent")) {
    std::string name(new_agent_name_.data());
    std::string role(new_agent_role_.data());
    if (name.empty()) {
      name = "Agent " + std::to_string(agents_.size() + 1);
    }
    if (role.empty()) role = "Generalist";

    AgentState agent;
    agent.name = name;
    agent.role = role;
    agent.status = "Active";
    agent.enabled = true;
    agent.data_backed = false;
    agent.queue_depth = 0;
    agent.tasks_completed = 0;
    agent.success_rate = 0.82f;
    agent.avg_latency_ms = 24.0f;
    agent.cpu_pct = 20.0f;
    agent.mem_pct = 20.0f;
    agent.activity_phase = 0.3f * static_cast<float>(agents_.size() + 1);
    agents_.push_back(std::move(agent));

    AppendLog(name, "Agent provisioned.", "agent");
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##SpawnAgentCount", &spawn_agent_count_);
  spawn_agent_count_ = std::max(1, std::min(spawn_agent_count_, 12));
  ImGui::SameLine();
  if (ImGui::Button("Spawn Batch")) {
    std::string base_name(new_agent_name_.data());
    if (base_name.empty()) base_name = "Agent";
    std::string role(new_agent_role_.data());
    if (role.empty()) role = "Generalist";

    for (int i = 0; i < spawn_agent_count_; ++i) {
      std::string name =
          base_name + " " + std::to_string(agents_.size() + 1);
      AgentState agent;
      agent.name = name;
      agent.role = role;
      agent.status = "Active";
      agent.enabled = true;
      agent.data_backed = false;
      agent.queue_depth = 0;
      agent.tasks_completed = 0;
      agent.success_rate = 0.8f;
      agent.avg_latency_ms = 26.0f;
      agent.cpu_pct = 18.0f;
      agent.mem_pct = 18.0f;
      agent.activity_phase = 0.3f * static_cast<float>(agents_.size() + 1);
      agents_.push_back(std::move(agent));
    }
    AppendLog("system", "Batch spawn complete.", "system");
  }

  if (ImGui::Button("Pause All")) {
    for (auto& agent : agents_) {
      agent.enabled = false;
      agent.status = "Paused";
    }
    AppendLog("system", "All agents paused.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Resume All")) {
    for (auto& agent : agents_) {
      agent.enabled = true;
      if (agent.status == "Paused") agent.status = "Idle";
    }
    AppendLog("system", "All agents resumed.", "system");
  }

  float table_height = ImGui::GetContentRegionAvail().y;
  if (ImGui::BeginTable("AgentsTable", 8,
                        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_ScrollY,
                        ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Agent");
    ImGui::TableSetupColumn("Role");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Queue");
    ImGui::TableSetupColumn("Success");
    ImGui::TableSetupColumn("Latency");
    ImGui::TableSetupColumn("CPU/Mem");
    ImGui::TableSetupColumn("On");
    ImGui::TableHeadersRow();

    for (size_t i = 0; i < agents_.size(); ++i) {
      auto& agent = agents_[i];
      ImGui::PushID(static_cast<int>(i));
      const char* status =
          agent.enabled ? (agent.queue_depth > 0 ? "Busy" : "Idle") : "Paused";

      ImGui::TableNextRow();
      bool is_selected = (selected_agent_index_ == static_cast<int>(i));
      
      // Pulse animation for active agents
      float pulse = 0.0f;
      if (use_pulse_animations_ && agent.enabled && agent.status != "Idle") {
          pulse = 0.5f + 0.5f * std::sin(pulse_timer_ * 6.0f);
      }
      
      if (pulse > 0.01f) {
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(GetStepColor(pulse * 0.2f)));
      }

      if (ImGui::Selectable(agent.name.c_str(), is_selected,
                            ImGuiSelectableFlags_SpanAllColumns |
                                 ImGuiSelectableFlags_AllowOverlap)) {
        selected_agent_index_ = static_cast<int>(i);
      }

      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", agent.role.c_str());
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%s", status);
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%d", agent.queue_depth);
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%.0f%%", agent.success_rate * 100.0f);
      ImGui::TableSetColumnIndex(5);
      ImGui::Text("%.1f ms", agent.avg_latency_ms);
      ImGui::TableSetColumnIndex(6);
      ImGui::Text("%.0f/%.0f", agent.cpu_pct, agent.mem_pct);
      ImGui::TableSetColumnIndex(7);
      ImGui::Checkbox("##enabled", &agent.enabled);
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  if (selected_agent_index_ >= 0 &&
      selected_agent_index_ < static_cast<int>(agents_.size())) {
    auto& agent = agents_[selected_agent_index_];
    ImGui::Separator();
    ImGui::Text("Agent Details: %s", agent.name.c_str());
    ImGui::Columns(2, "AgentDetailCols", false);
    ImGui::Text("Role: %s", agent.role.c_str());
    ImGui::Text("Status: %s", agent.status.c_str());
    ImGui::Text("Tasks: %d", agent.tasks_completed);
    ImGui::NextColumn();
    
    // Sparkline simulation
    if (sparkline_data_.size() < 40) {
       for(int i=0; i<40; ++i) sparkline_data_.push_back(0.4f + 0.5f * (float)rand()/RAND_MAX);
    }
    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
    if (ImPlot::BeginPlot("##Sparkline", ImVec2(-1, 60), ImPlotFlags_CanvasOnly)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::PlotLine("Activity", sparkline_data_.data(), 40);
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor();
    ImGui::Columns(1);
    
    if (ImGui::Button("Reset Agent Stats")) {
        agent.tasks_completed = 0;
        AppendLog(agent.name, "Stats reset.", "agent");
    }
    ImGui::SameLine();
    if (ImGui::Button("Force Restart")) {
        AppendLog(agent.name, "Restarting...", "system");
    }
  }
}

void App::RenderMissionsTab() {
  ImGui::Text("Mission Queue");
  ImGui::InputTextWithHint("##MissionName", "Mission name",
                           new_mission_name_.data(), new_mission_name_.size());
  ImGui::InputTextWithHint("##MissionOwner", "Owner",
                           new_mission_owner_.data(), new_mission_owner_.size());
  ImGui::SliderInt("Priority", &new_mission_priority_, 1, 5);

  if (ImGui::Button("Create Mission")) {
    std::string name(new_mission_name_.data());
    std::string owner(new_mission_owner_.data());
    if (name.empty()) {
      name = "Mission " + std::to_string(missions_.size() + 1);
    }
    if (owner.empty()) owner = "Ops";

    MissionState mission;
    mission.name = name;
    mission.owner = owner;
    mission.status = "Queued";
    mission.data_backed = false;
    mission.priority = new_mission_priority_;
    mission.progress = 0.0f;
    missions_.push_back(std::move(mission));
    AppendLog("system", "Mission queued: " + name, "system");
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##SpawnMissionCount", &spawn_mission_count_);
  spawn_mission_count_ = std::max(1, std::min(spawn_mission_count_, 10));
  ImGui::SameLine();
  if (ImGui::Button("Spawn Batch")) {
    std::string base_name(new_mission_name_.data());
    if (base_name.empty()) base_name = "Mission";
    std::string owner(new_mission_owner_.data());
    if (owner.empty()) owner = "Ops";

    for (int i = 0; i < spawn_mission_count_; ++i) {
      MissionState mission;
      mission.name = base_name + " " + std::to_string(missions_.size() + 1);
      mission.owner = owner;
      mission.status = "Queued";
      mission.data_backed = false;
      mission.priority = new_mission_priority_;
      mission.progress = 0.0f;
      missions_.push_back(std::move(mission));
    }
    AppendLog("system", "Batch missions queued.", "system");
  }

  if (ImGui::Button("Clear Completed")) {
    missions_.erase(std::remove_if(missions_.begin(), missions_.end(),
                                   [](const MissionState& mission) {
                                     return mission.status == "Complete" &&
                                            !mission.data_backed;
                                   }),
                    missions_.end());
  }

  float table_height = ImGui::GetContentRegionAvail().y;
  if (ImGui::BeginTable("MissionsTable", 6,
                        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_ScrollY,
                        ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Mission");
    ImGui::TableSetupColumn("Owner");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Priority");
    ImGui::TableSetupColumn("Progress");
    ImGui::TableSetupColumn("Data");
    ImGui::TableHeadersRow();

    for (size_t i = 0; i < missions_.size(); ++i) {
      auto& mission = missions_[i];
      ImGui::PushID(static_cast<int>(i));
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s", mission.name.c_str());
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", mission.owner.c_str());
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%s", mission.status.c_str());
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%d", mission.priority);
      ImGui::TableSetColumnIndex(4);
      ImGui::ProgressBar(mission.progress, ImVec2(-FLT_MIN, 0.0f));
      ImGui::TableSetColumnIndex(5);
      ImGui::Text("%s", mission.data_backed ? "data" : "live");
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::Text("System Actions");
  if (ImGui::Button("Trigger Training Hub")) {
      AppendLog("system", "Training Hub run requested.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Run Evaluation Loop")) {
      AppendLog("system", "Evaluation Loop triggered.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Rebuild Knowledge Index")) {
      AppendLog("system", "Knowledge Index rebuild started.", "system");
  }
}

void App::RenderLogsTab() {
  ImGui::Text("Logs & Agent Chat");
  ImGui::InputTextWithHint("##LogFilter", "Filter logs",
                           log_filter_.data(), log_filter_.size());

  std::vector<const char*> agent_labels;
  agent_labels.push_back("All");
  for (const auto& agent : agents_) {
    agent_labels.push_back(agent.name.c_str());
  }
  if (log_agent_index_ < 0 ||
      log_agent_index_ >= static_cast<int>(agent_labels.size())) {
    log_agent_index_ = 0;
  }
  ImGui::Combo("Target", &log_agent_index_, agent_labels.data(),
               static_cast<int>(agent_labels.size()));

  float log_height = ImGui::GetContentRegionAvail().y - 60.0f;
  if (log_height < 120.0f) log_height = 120.0f;
  ImGui::BeginChild("LogList", ImVec2(0, log_height), true,
                    ImGuiWindowFlags_AlwaysVerticalScrollbar);

  std::string filter(log_filter_.data());
  for (const auto& entry : logs_) {
    if (log_agent_index_ > 0 &&
        entry.agent != agent_labels[log_agent_index_]) {
      continue;
    }
    if (!filter.empty()) {
      if (entry.message.find(filter) == std::string::npos &&
          entry.agent.find(filter) == std::string::npos) {
        continue;
      }
    }

    ImVec4 color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    if (entry.kind == "system") {
      color = ImVec4(0.65f, 0.80f, 0.95f, 1.0f);
    } else if (entry.kind == "user") {
      color = ImVec4(0.95f, 0.85f, 0.55f, 1.0f);
    }

    ImGui::TextColored(color, "[%s] %s", entry.agent.c_str(),
                       entry.message.c_str());
  }
  ImGui::EndChild();

  ImGui::InputTextWithHint("##ChatInput", "Talk to agent...",
                           chat_input_.data(), chat_input_.size());
  ImGui::SameLine();
  if (ImGui::Button("Send")) {
    std::string message(chat_input_.data());
    if (!message.empty()) {
      std::string target = agent_labels[log_agent_index_];
      AppendLog("user", "To " + target + ": " + message, "user");
      if (log_agent_index_ > 0) {
        AppendLog(target, "Acknowledged: " + message, "agent");
      }
      chat_input_[0] = '\0';
    }
  }
}

void App::RenderContextTab() {
  ImGui::Text("AFS Context Browser");
  ImGui::Separator();

  // Navigation
  if (ImGui::Button("..") || ImGui::Button("Up")) {
    if (current_browser_path_.has_parent_path()) {
      current_browser_path_ = current_browser_path_.parent_path();
      browser_entries_.clear();
    }
  }
  ImGui::SameLine();
  ImGui::TextDisabled("Path: %s", current_browser_path_.string().c_str());

  // Refresh entries if empty
  if (browser_entries_.empty()) {
    try {
      for (const auto& entry : std::filesystem::directory_iterator(current_browser_path_)) {
        FileEntry fe;
        fe.name = entry.path().filename().string();
        fe.path = entry.path();
        fe.is_directory = entry.is_directory();
        fe.size = fe.is_directory ? 0 : entry.file_size();
        browser_entries_.push_back(fe);
      }
      std::sort(browser_entries_.begin(), browser_entries_.end(), [](const FileEntry& a, const FileEntry& b) {
        if (a.is_directory != b.is_directory) return a.is_directory;
        return a.name < b.name;
      });
    } catch (...) {}
  }

  float table_height = ImGui::GetContentRegionAvail().y * 0.6f;
  if (ImGui::BeginTable("FileBrowser", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableHeadersRow();

    for (const auto& entry : browser_entries_) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (entry.is_directory) {
        if (ImGui::Selectable((entry.name + "/").c_str(), false)) {
          current_browser_path_ = entry.path;
          browser_entries_.clear();
          break;
        }
      } else {
        ImGui::Text("%s", entry.name.c_str());
      }

      ImGui::TableNextColumn();
      if (!entry.is_directory) {
        ImGui::TextDisabled("%.1f KB", entry.size / 1024.0f);
      }

      ImGui::TableNextColumn();
      if (!entry.is_directory) {
        if (ImGui::Button(("Add##" + entry.name).c_str())) {
          ContextItem item;
          item.name = entry.name;
          item.path = entry.path;
          item.type = entry.path.extension().string();
          selected_context_.push_back(item);
          AppendLog("system", "Added to context: " + entry.name, "system");
        }
      }
    }
    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::Text("Selected Context (%d items)", (int)selected_context_.size());
  if (ImGui::BeginChild("ContextList", ImVec2(0, 0), true)) {
    for (size_t i = 0; i < selected_context_.size(); ++i) {
      auto& item = selected_context_[i];
      ImGui::PushID((int)i);
      ImGui::Checkbox("##on", &item.enabled);
      ImGui::SameLine();
      ImGui::Text("%s", item.name.c_str());
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", item.path.string().c_str());
      ImGui::SameLine(ImGui::GetContentRegionAvail().x - 40);
      if (ImGui::Button("X")) {
        selected_context_.erase(selected_context_.begin() + i);
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
    ImGui::EndChild();
  }
}

void App::RenderAgentPromptTab() {
  ImGui::Text("Agent Orchestrator Prompt");
  ImGui::Separator();

  ImGui::Text("System Prompt");
  ImGui::InputTextMultiline("##SysPrompt", system_prompt_.data(), system_prompt_.size(), ImVec2(-1, 80));

  ImGui::Spacing();
  ImGui::Text("User Message");
  ImGui::InputTextMultiline("##UserPrompt", user_prompt_.data(), user_prompt_.size(), ImVec2(-1, 120));

  static int target_agent = 0;
  std::vector<const char*> agent_names = {"Orchestrator", "Coordinator"};
  for (const auto& a : agents_) agent_names.push_back(a.name.c_str());
  
  ImGui::Combo("Target", &target_agent, agent_names.data(), (int)agent_names.size());

  if (ImGui::Button("Trigger Background Agent", ImVec2(-1, 40))) {
    std::string msg = "Sent prompt to " + std::string(agent_names[target_agent]);
    AppendLog("user", msg, "user");
    AppendLog(agent_names[target_agent], "Analyzing context with " + std::to_string(selected_context_.size()) + " files...", "agent");
    user_prompt_[0] = '\0';
  }

  ImGui::Separator();
  ImGui::TextDisabled("Quick Context Meta:");
  for (const auto& item : selected_context_) {
    if (item.enabled) {
      ImGui::TextDisabled(" - %s (%s)", item.name.c_str(), item.type.c_str());
    }
  }
}

void App::RenderTablesTab() {
  if (ImGui::BeginTabBar("TableGroups")) {
    if (ImGui::BeginTabItem("Generator Detailed")) {
      const auto& stats = loader_.GetGeneratorStats();
      if (ImGui::BeginTable("GenDetailed", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Generator");
        ImGui::TableSetupColumn("Accepted");
        ImGui::TableSetupColumn("Rejected");
        ImGui::TableSetupColumn("Rate %");
        ImGui::TableSetupColumn("Avg Q");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        for (const auto& s : stats) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn(); ImGui::Text("%s", s.name.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%d", s.samples_accepted);
          ImGui::TableNextColumn(); ImGui::Text("%d", s.samples_rejected);
          ImGui::TableNextColumn(); 
          float rate = s.acceptance_rate * 100.0f;
          ImGui::Text("%.1f%%", rate);
          if (rate < 40.0f) ImGui::SameLine(); if (rate < 40.0f) ImGui::TextColored(ImVec4(1,0,0,1), "[!] ");
          
          ImGui::TableNextColumn(); ImGui::Text("%.3f", s.avg_quality);
          ImGui::TableNextColumn(); 
          if (s.samples_rejected > s.samples_accepted) ImGui::TextColored(ImVec4(1,0.5,0,1), "Struggling");
          else ImGui::TextColored(ImVec4(0,1,0.5,1), "Healthy");
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }
    
    if (ImGui::BeginTabItem("Quality Metrics")) {
      const auto& trends = loader_.GetQualityTrends();
      if (ImGui::BeginTable("QualityDetailed", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Domain");
        ImGui::TableSetupColumn("Metric");
        ImGui::TableSetupColumn("Mean Score");
        ImGui::TableSetupColumn("Trend");
        ImGui::TableSetupColumn("Sparkline");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < trends.size(); ++i) {
          const auto& t = trends[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn(); ImGui::Text("%s", t.domain.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%s", t.metric.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%.4f", t.mean);
          ImGui::TableNextColumn(); 
          if (t.trend_direction == "improving") ImGui::TextColored(ImVec4(0,1,0,1), "INC");
          else if (t.trend_direction == "declining") ImGui::TextColored(ImVec4(1,0,0,1), "DEC");
          else ImGui::Text("---");
          
          ImGui::TableNextColumn();
          ImGui::PushID((int)i);
          if (ImPlot::BeginPlot("##Spark", ImVec2(-1, 24), ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs)) {
            ImPlot::SetupAxes(nullptr,nullptr,ImPlotAxisFlags_NoDecorations,ImPlotAxisFlags_NoDecorations);
            ImPlot::PlotLine("##v", t.values.data(), (int)t.values.size());
            ImPlot::EndPlot();
          }
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Training Runs")) {
        const auto& runs = loader_.GetTrainingRuns();
        if (ImGui::BeginTable("TrainingDetailed", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("Run ID");
            ImGui::TableSetupColumn("Model");
            ImGui::TableSetupColumn("Samples");
            ImGui::TableSetupColumn("Loss");
            ImGui::TableSetupColumn("Eval Metrics");
            ImGui::TableHeadersRow();

            for (const auto& r : runs) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%s", r.run_id.substr(0, 16).c_str());
                ImGui::TableNextColumn(); ImGui::Text("%s", r.model_name.c_str());
                ImGui::TableNextColumn(); ImGui::Text("%d", r.samples_count);
                ImGui::TableNextColumn(); ImGui::Text("%.5f", r.final_loss);
                ImGui::TableNextColumn();
                for (const auto& [name, val] : r.eval_metrics) {
                    ImGui::TextDisabled("%s: %.3f", name.c_str(), val);
                }
            }
            ImGui::EndTable();
        }
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void App::RenderQualityChart() {
  RenderChartHeader("QUALITY TRENDS", "Displays model performance metrics across active training domains. Solid lines indicate scores; shaded area indicates the Optimal Strategy Zone (>0.85).");

  const auto& trends = loader_.GetQualityTrends();
  if (trends.empty()) {
    ImGui::TextDisabled("No quality trend data available");
    return;
  }

  if (ImPlot::BeginPlot("##QualityTrends", ImGui::GetContentRegionAvail(), ImPlotFlags_NoLegend)) {
    ApplyPremiumPlotStyles("##QualityTrends");
    ImPlot::SetupAxes("Time Step", "Score (0-1)");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.1, ImPlotCond_Always);
    
    // Help markers and goal regions...
    double goal_x[2] = {-100, 1000};
    double goal_y1[2] = {0.85, 0.85};
    double goal_y2[2] = {1.1, 1.1};
    ImPlot::SetNextFillStyle(ImVec4(0, 1, 0, 0.05f));
    ImPlot::PlotShaded("Goal Region", goal_x, goal_y1, goal_y2, 2);
    
    ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 0.4f), 1.0f);
    ImPlot::PlotLine("Requirement (0.85)", goal_x, goal_y1, 2);

    for (const auto& trend : trends) {
      if (trend.values.empty()) continue;
      std::string label = trend.domain + " (" + trend.metric + ")";
      ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.15f);
      ImPlot::PlotLine(label.c_str(), trend.values.data(), (int)trend.values.size());
      
      // Inline label if hovering peak
      if (ImPlot::IsPlotHovered()) {
          ImPlotPoint mouse = ImPlot::GetPlotMousePos();
          // Logic for tooltips/hover would go here if needed
      }
    }
    
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderGeneratorChart() {
  RenderChartHeader("GENERATOR EFFICIENCY", "Acceptance rates for active data generators. Rates < 40% (Warning Zone) indicate generators struggling with current model constraints.");

  const auto& stats = loader_.GetGeneratorStats();
  if (stats.empty()) {
    ImGui::TextDisabled("No generator stats available");
    return;
  }
  
  // ... (label preparation)
  std::vector<const char*> labels;
  std::vector<float> rates;
  std::vector<std::string> label_storage;
  for (const auto& s : stats) {
    std::string name = s.name;
    size_t pos = name.find("DataGenerator");
    if (pos != std::string::npos) name = name.substr(0, pos);
    label_storage.push_back(name);
    rates.push_back(s.acceptance_rate * 100.0f);
  }
  for (const auto& s : label_storage) labels.push_back(s.c_str());

  if (ImPlot::BeginPlot("##GeneratorStats", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Generator", "Acceptance %");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());
    ApplyPremiumPlotStyles("##GeneratorStats");
    
    // Warning Zone Overlay
    double wx[2] = {-1, 100};
    double wy1[2] = {0, 0};
    double wy2[2] = {40, 40};
    ImPlot::SetNextFillStyle(ImVec4(1, 0, 0, 0.1f));
    ImPlot::PlotShaded("Low Efficiency", wx, wy1, wy2, 2);

    ImPlot::PlotBars("Rate", rates.data(), static_cast<int>(rates.size()), 0.67);
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderCoverageChart() {
  RenderChartHeader("DENSITY COVERAGE", "Displays sample counts across latent space regions. Sparse regions (<50% of avg) indicate under-sampled scenarios.");

  const auto& regions = loader_.GetEmbeddingRegions();
  const auto& coverage = loader_.GetCoverage();

  if (regions.empty()) {
    ImGui::TextDisabled("No embedding coverage data available");
    return;
  }

  // Scatter plot of region densities
  std::vector<float> xs, dense_x, dense_y, sparse_x, sparse_y;
  float total = 0.0f;
  for (const auto& r : regions) total += static_cast<float>(r.sample_count);
  float avg = total / static_cast<float>(regions.size());

  for (size_t i = 0; i < regions.size(); ++i) {
    float x = static_cast<float>(i);
    float y = static_cast<float>(regions[i].sample_count);
    if (y < avg * 0.5f) {
      sparse_x.push_back(x);
      sparse_y.push_back(y);
    } else {
      dense_x.push_back(x);
      dense_y.push_back(y);
    }
  }

  if (ImPlot::BeginPlot("##Coverage", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Region Index", "Samples");
    ApplyPremiumPlotStyles("##Coverage");
    
    // Low Density Zone Overlay
    double lx[2] = {-10, static_cast<double>(regions.size() + 10)};
    double ly1[2] = {0, 0};
    double ly2[2] = {avg * 0.5, avg * 0.5};
    ImPlot::SetNextFillStyle(ImVec4(1, 0.5f, 0, 0.1f));
    ImPlot::PlotShaded("Sparse Zone", lx, ly1, ly2, 2);

    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, GetThemeColor(ImGuiCol_PlotLines));
    ImPlot::PlotScatter("Healthy", dense_x.data(), dense_y.data(), (int)dense_x.size());
    
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(1, 0.4f, 0, 1));
    ImPlot::PlotScatter("At Risk", sparse_x.data(), sparse_y.data(), (int)sparse_x.size());
    
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderTrainingChart() {
  auto runs = loader_.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training run data available");
    return;
  }

  // Sort by loss (ascending)
  std::sort(runs.begin(), runs.end(),
            [](const TrainingRunData& a, const TrainingRunData& b) {
              return a.final_loss < b.final_loss;
            });

  // Limit to top 10
  if (runs.size() > 10) runs.resize(10);

  std::vector<const char*> labels;
  std::vector<float> losses;
  std::vector<std::string> label_storage;

  for (const auto& r : runs) {
    // Truncate run ID for display
    std::string id = r.run_id.substr(0, std::min(r.run_id.size(), size_t(12)));
    label_storage.push_back(id);
    losses.push_back(r.final_loss);
  }

  for (const auto& s : label_storage) {
    labels.push_back(s.c_str());
  }

  if (ImPlot::BeginPlot("##TrainingLoss", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Run", "Final Loss");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());

    ImPlot::PlotBars("Loss", losses.data(), static_cast<int>(losses.size()), 0.67);

    ImPlot::EndPlot();
  }
}

void App::RenderTrainingLossChart() {
  const auto& runs = loader_.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training run data available");
    return;
  }

  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(runs.size());
  ys.reserve(runs.size());
  for (const auto& run : runs) {
    xs.push_back(static_cast<float>(run.samples_count));
    ys.push_back(run.final_loss);
  }

  if (ImPlot::BeginPlot("##LossVsSamples", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Samples", "Final Loss");
    ImPlot::PlotScatter("Runs", xs.data(), ys.data(),
                        static_cast<int>(xs.size()));
    ImPlot::EndPlot();
  }
}

void App::RenderDomainCoverageChart() {
  const auto& coverage = loader_.GetCoverage();
  if (coverage.domain_coverage.empty()) {
    ImGui::TextDisabled("No domain coverage data available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;

  for (const auto& [domain, value] : coverage.domain_coverage) {
    label_storage.push_back(domain);
    values.push_back(value * 100.0f);
  }

  for (const auto& label : label_storage) {
    labels.push_back(label.c_str());
  }

  if (ImPlot::BeginPlot("##DomainCoverage", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Domain", "Coverage %");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());
    ImPlot::PlotBars("Coverage", values.data(), static_cast<int>(values.size()),
                     0.67);
    ImPlot::EndPlot();
  }
}

void App::RenderEmbeddingQualityChart() {
  const auto& regions = loader_.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding quality data available");
    return;
  }

  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(regions.size());
  ys.reserve(regions.size());
  for (const auto& region : regions) {
    xs.push_back(static_cast<float>(region.index));
    ys.push_back(region.avg_quality);
  }

  if (ImPlot::BeginPlot("##EmbeddingQuality", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Region", "Avg Quality");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
    ImPlot::PlotScatter("Quality", xs.data(), ys.data(),
                        static_cast<int>(xs.size()));
    ImPlot::EndPlot();
  }
}

void App::RenderAgentThroughputChart() {
  RenderChartHeader("AGENT THROUGHPUT", "Real-time task processing rate across the swarm. Higher peaks indicate high-availability periods; the dashed line represents the Swarm Target (1.0k).");

  if (agents_.empty()) {
    ImGui::TextDisabled("Data stream offline");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> tasks;
  std::vector<float> queues;
  std::vector<std::string> label_storage;

  for (size_t i = 0; i < agents_.size(); ++i) {
    const auto& agent = agents_[i];
    // Simplify labels to prevent overlap: "A1 (ID)", etc.
    char buf[16];
    snprintf(buf, sizeof(buf), "A%zu", i + 1);
    label_storage.push_back(buf);
    
    tasks.push_back(static_cast<float>(agent.tasks_completed));
    queues.push_back(static_cast<float>(agent.queue_depth));
  }

  for (const auto& label : label_storage) {
    labels.push_back(label.c_str());
  }

  if (ImPlot::BeginPlot("##AgentThroughput", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Agent Index", "Total Tasks Completed");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 5000, ImGuiCond_Once); // Initial guess
    
    if (!labels.empty()) {
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                               static_cast<int>(labels.size()), labels.data());
    }
    
    ApplyPremiumPlotStyles("##Throughput");
    
    ImPlot::PlotBars("Tasks Completed", tasks.data(), static_cast<int>(tasks.size()), 0.6);
    ImPlot::PlotLine("Queue Depth", queues.data(), static_cast<int>(queues.size()));
    
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderMissionQueueChart() {
  if (missions_.empty()) {
    ImGui::TextDisabled("No mission data available");
    return;
  }

  int queued = 0;
  int active = 0;
  int complete = 0;
  int other = 0;

  for (const auto& mission : missions_) {
    if (mission.status == "Queued") {
      ++queued;
    } else if (mission.status == "Active") {
      ++active;
    } else if (mission.status == "Complete") {
      ++complete;
    } else {
      ++other;
    }
  }

  std::array<const char*, 4> labels = {"Queued", "Active", "Complete", "Other"};
  std::array<float, 4> values = {static_cast<float>(queued),
                                 static_cast<float>(active),
                                 static_cast<float>(complete),
                                 static_cast<float>(other)};

  if (ImPlot::BeginPlot("##MissionQueue", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Status", "Count");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    ImPlot::PlotBars("Missions", values.data(), 4, 0.67);
    ImPlot::EndPlot();
  }
}

void App::RenderQualityDirectionChart() {
  RenderChartHeader("QUALITY TRENDS", "Overview of quality trend directions across all tracked metrics. 'Sparse' indicates insufficient data for a reliable trend.");

  const auto& trends = loader_.GetQualityTrends();
  if (trends.empty()) {
    ImGui::TextDisabled("No quality trend data available");
    return;
  }

  int improving = 0;
  int declining = 0;
  int stable = 0;
  int insufficient = 0;

  for (const auto& trend : trends) {
    if (trend.trend_direction == "improving") {
      ++improving;
    } else if (trend.trend_direction == "declining") {
      ++declining;
    } else if (trend.trend_direction == "stable") {
      ++stable;
    } else {
      ++insufficient;
    }
  }

  std::array<const char*, 4> labels = {"Improving", "Stable", "Declining",
                                       "Sparse"};
  std::array<float, 4> values = {static_cast<float>(improving),
                                 static_cast<float>(stable),
                                 static_cast<float>(declining),
                                 static_cast<float>(insufficient)};

  if (ImPlot::BeginPlot("##QualityDirection", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Trend", "Count");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    ApplyPremiumPlotStyles("##QualityDirection");
    
    ImPlot::PlotBars("Trends", values.data(), 4, 0.67);
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderGeneratorMixChart() {
  const auto& stats = loader_.GetGeneratorStats();
  if (stats.empty()) {
    ImGui::TextDisabled("No generator stats available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> accepted;
  std::vector<float> rejected;
  std::vector<float> quality;
  std::vector<float> xs;
  std::vector<std::string> label_storage;

  for (const auto& s : stats) {
    std::string name = s.name;
    size_t pos = name.find("DataGenerator");
    if (pos != std::string::npos) {
      name = name.substr(0, pos);
    }
    label_storage.push_back(name);
    accepted.push_back(static_cast<float>(s.samples_accepted));
    rejected.push_back(static_cast<float>(s.samples_rejected));
    quality.push_back(Clamp01(s.avg_quality) * 100.0f);
  }

  for (size_t i = 0; i < label_storage.size(); ++i) {
    labels.push_back(label_storage[i].c_str());
    xs.push_back(static_cast<float>(i));
  }

  if (ImPlot::BeginPlot("##GeneratorMix", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Generator", "Samples");
    ImPlot::SetupAxis(ImAxis_Y2, "Avg Quality %");
    ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, 100.0, ImPlotCond_Once);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());

    ImPlot::PlotBars("Accepted", accepted.data(),
                     static_cast<int>(accepted.size()), 0.35, -0.2);
    ImPlot::PlotBars("Rejected", rejected.data(),
                     static_cast<int>(rejected.size()), 0.35, 0.2);

    ImPlot::SetAxis(ImAxis_Y2);
    ImPlot::PlotLine("Avg Quality %", xs.data(), quality.data(),
                     static_cast<int>(quality.size()));
    ImPlot::SetAxis(ImAxis_Y1);

    ImPlot::EndPlot();
  }
}

void App::RenderEmbeddingDensityChart() {
  const auto& regions = loader_.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding density data available");
    return;
  }

  int min_count = regions.front().sample_count;
  int max_count = regions.front().sample_count;
  for (const auto& region : regions) {
    min_count = std::min(min_count, region.sample_count);
    max_count = std::max(max_count, region.sample_count);
  }

  const int bins = 8;
  std::vector<float> counts(bins, 0.0f);
  int range = std::max(1, max_count - min_count);
  float bin_size = static_cast<float>(range) / static_cast<float>(bins);

  for (const auto& region : regions) {
    int idx = static_cast<int>(
        (static_cast<float>(region.sample_count - min_count)) / bin_size);
    if (idx >= bins) idx = bins - 1;
    counts[idx] += 1.0f;
  }

  std::vector<const char*> labels;
  std::vector<std::string> label_storage;
  label_storage.reserve(bins);
  for (int i = 0; i < bins; ++i) {
    int start = static_cast<int>(min_count + bin_size * i);
    int end = static_cast<int>(min_count + bin_size * (i + 1));
    label_storage.push_back(std::to_string(start) + "-" +
                            std::to_string(end));
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  if (ImPlot::BeginPlot("##EmbeddingDensity", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Samples", "Regions");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(bins - 1), bins,
                           labels.data());
    ImPlot::PlotBars("Regions", counts.data(), bins, 0.67);
    ImPlot::EndPlot();
  }
}

void App::RenderAgentUtilizationChart() {
  if (agents_.empty()) {
    ImGui::TextDisabled("No agent utilization data available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> cpu;
  std::vector<float> mem;
  std::vector<std::string> label_storage;

  for (const auto& agent : agents_) {
    label_storage.push_back(agent.name);
    cpu.push_back(agent.cpu_pct);
    mem.push_back(agent.mem_pct);
  }

  for (const auto& label : label_storage) labels.push_back(label.c_str());

  if (ImPlot::BeginPlot("##AgentUtil", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Agent", "Utilization %");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());
    ImPlot::PlotBars("CPU", cpu.data(), static_cast<int>(cpu.size()), 0.35, -0.2);
    ImPlot::PlotBars("Mem", mem.data(), static_cast<int>(mem.size()), 0.35, 0.2);
    ImPlot::EndPlot();
  }
}

void App::RenderMissionProgressChart() {
  if (missions_.empty()) {
    ImGui::TextDisabled("No mission progress data available");
    return;
  }

  std::array<float, 4> buckets = {0.0f, 0.0f, 0.0f, 0.0f};
  for (const auto& mission : missions_) {
    float progress = Clamp01(mission.progress);
    if (progress < 0.25f) {
      buckets[0] += 1.0f;
    } else if (progress < 0.5f) {
      buckets[1] += 1.0f;
    } else if (progress < 0.75f) {
      buckets[2] += 1.0f;
    } else {
      buckets[3] += 1.0f;
    }
  }

  std::array<const char*, 4> labels = {"0-25%", "25-50%", "50-75%", "75-100%"};
  if (ImPlot::BeginPlot("##MissionProgress", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Progress", "Missions");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    ImPlot::PlotBars("Missions", buckets.data(), 4, 0.67);
    ImPlot::EndPlot();
  }
}

void App::RenderEvalMetricsChart() {
  const auto& runs = loader_.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training metrics available");
    return;
  }

  std::map<std::string, std::pair<float, int>> accum;
  for (const auto& run : runs) {
    for (const auto& [metric, value] : run.eval_metrics) {
      auto& slot = accum[metric];
      slot.first += value;
      slot.second += 1;
    }
  }

  if (accum.empty()) {
    ImGui::TextDisabled("No eval metrics reported");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;
  for (const auto& [metric, data] : accum) {
    label_storage.push_back(metric);
    values.push_back(data.first / static_cast<float>(std::max(1, data.second)));
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  if (ImPlot::BeginPlot("##EvalMetrics", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Metric", "Avg");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());
    ImPlot::PlotBars("Avg", values.data(), static_cast<int>(values.size()), 0.67);
    ImPlot::EndPlot();
  }
}

void App::RenderRejectionChart() {
  RenderChartHeader("REJECTION REASONS", "Top reasons for sample rejection. High counts in specific categories may indicate issues with data generation or filtering.");

  const auto& summary = loader_.GetRejectionSummary();
  if (summary.reasons.empty()) {
    ImGui::TextDisabled("No rejection data available");
    return;
  }

  // Sort by count and take top 8
  std::vector<std::pair<std::string, int>> sorted(summary.reasons.begin(),
                                                   summary.reasons.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  if (sorted.size() > 8) sorted.resize(8);

  std::vector<const char*> labels;
  std::vector<float> counts;
  std::vector<std::string> label_storage;

  for (const auto& [reason, count] : sorted) {
    // Format reason for display
    std::string formatted = reason;
    std::replace(formatted.begin(), formatted.end(), '_', ' ');
    label_storage.push_back(formatted);
    counts.push_back(static_cast<float>(count));
  }

  for (const auto& s : label_storage) {
    labels.push_back(s.c_str());
  }

  if (ImPlot::BeginPlot("##Rejections", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Reason", "Count");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());
    ApplyPremiumPlotStyles("##Rejections");

    ImPlot::PlotBars("Count", counts.data(), static_cast<int>(counts.size()),
                     0.67);
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderServicesTab() {
  ImGui::Text("Core Services");
  
  auto render_service = [this](const char* name, const char* status, float health, const char* desc) {
    ImGui::PushID(name);
    ImGui::BeginGroup();
    ImGui::Text("%s", name);
    ImGui::TextDisabled("%s", desc);
    
    ImVec4 status_color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);
    if (strcmp(status, "Warning") == 0) status_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
    else if (strcmp(status, "Error") == 0) status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
    
    ImGui::TextColored(status_color, "Status: %s", status);
    ImGui::ProgressBar(health, ImVec2(-1.0f, 0.0f));
    ImGui::EndGroup();
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
  };

  render_service("Orchestrator", "Active", 0.98f, "Main mission control & dispatch");
  render_service("Knowledge Base", "Active", 0.92f, "Vector DB & Concept Index");
  render_service("Trainer", "Active", 0.85f, "Active learning loop coordination");
  render_service("Embedding SVC", "Warning", 0.65f, "Latency spike detected in region 42");
  
  if (ImGui::Button("Reset Services")) {
    AppendLog("system", "Services reset signal sent.", "system");
  }
}

void App::RenderKnowledgeGraph() {
  if (knowledge_concepts_.empty()) return;

  if (ImPlot::BeginPlot("##KnowledgeGraph", ImGui::GetContentRegionAvail(), ImPlotFlags_NoMouseText)) {
    // Setup First
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, 100);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);

    // Render Edges
    ImVec4 edge_color = GetThemeColor(ImGuiCol_Text);
    edge_color.w = 0.15f;
    for (const auto& edge : knowledge_edges_) {
        float ex[2] = {knowledge_nodes_x_[edge.from], knowledge_nodes_x_[edge.to]};
        float ey[2] = {knowledge_nodes_y_[edge.from], knowledge_nodes_y_[edge.to]};
        ImPlot::SetNextLineStyle(edge_color, 1.5f);
        ImPlot::PlotLine("##edge", ex, ey, 2);
    }

    // Render Nodes
    ImVec4 node_color = GetThemeColor(ImGuiCol_PlotLines);
    for (size_t i = 0; i < knowledge_concepts_.size(); ++i) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, node_color, 1.0f, ImVec4(1,1,1,1));
      ImPlot::PlotScatter(knowledge_concepts_[i].c_str(), &knowledge_nodes_x_[i], &knowledge_nodes_y_[i], 1);
    }

    ImPlot::EndPlot();
  }
}

void App::RenderLatentSpaceChart() {
  RenderChartHeader("LATENT TOPOLOGY", "Visualization of the manifold learned by the model. Clusters indicate stable concept representations; voids represent potential logic gaps.");

  const auto& regions = loader_.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding data");
    return;
  }

  if (ImPlot::BeginPlot("##LatentSpace", ImGui::GetContentRegionAvail(), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
    ApplyPremiumPlotStyles("##LatentSpace");
    
    // Multicolored Cluster Trajectories
    std::vector<float> xs, ys;
    for (const auto& r : regions) {
      float angle = (float)r.index * 0.15f;
      float dist = 2.0f + (float)std::sin(r.index * 0.3f) * 1.5f;
      xs.push_back(std::cos(angle) * dist);
      ys.push_back(std::sin(angle) * dist);
    }

    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, GetThemeColor(ImGuiCol_PlotLines));
    ImPlot::PlotScatter("Embeddings", xs.data(), ys.data(), static_cast<int>(xs.size()));
    
    // Add anomaly highlight if a region is critically sparse
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].sample_count < 20) { // Arbitrary critical threshold
             ImPlot::Annotation(xs[i], ys[i], ImVec4(1,0,0,1), ImVec2(10,-10), true, "DENSITY DROP");
             break; // Just show one
        }
    }

    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderStatusBar() {
  ImGui::Separator();
  if (ImGui::BeginTable("StatusStrip", 2,
                        ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableNextColumn();
    if (loader_.HasData()) {
      ImGui::Text("Generators: %zu  |  Regions: %zu  |  Runs: %zu  |  F5 to refresh",
                  loader_.GetGeneratorStats().size(),
                  loader_.GetEmbeddingRegions().size(),
                  loader_.GetTrainingRuns().size());
    } else {
      ImGui::TextDisabled("No data loaded - Press F5 to refresh");
    }

    ImGui::TableNextColumn();
    ImGui::Text("Data: %s", data_path_.c_str());
    ImGui::EndTable();
  }
}

void App::RenderEffectivenessChart() {
  RenderChartHeader("DOMAIN TRAINING EFFECTIVENESS", "Measures the impact of training samples on model performance within specific domains. High effectiveness (>8.0) indicates high-value training data.");

  const auto& opt = loader_.GetOptimizationData();
  if (opt.domain_effectiveness.empty()) {
    ImGui::TextDisabled("No effectiveness data available.");
    return;
  }

  std::vector<const char*> labels;
  std::vector<double> values;
  for (const auto& [domain, val] : opt.domain_effectiveness) {
    labels.push_back(domain.c_str());
    values.push_back(val * 10.0);
  }

  if (ImPlot::BeginPlot("##Effectiveness", ImGui::GetContentRegionAvail())) {
    ImPlot::SetupAxes("Domain", "Effectiveness Score");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 12, ImPlotCond_Once);
    if (!values.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, values.size() - 1, values.size(), labels.data());
    }
    ApplyPremiumPlotStyles("##Effectiveness");
    
    // High Impact Area Overlay
    double tx[2] = {-1, 100};
    double ty1[2] = {8.0, 8.0};
    double ty2[2] = {12.0, 12.0};
    ImPlot::SetNextFillStyle(ImVec4(0, 1, 0, 0.08f));
    ImPlot::PlotShaded("High Impact", tx, ty1, ty2, 2);

    if (!values.empty()) {
      ImPlot::PlotBars("Effectiveness", values.data(), values.size(), 0.6);
    }
    
    ImPlot::PopStyleColor(2);
    ImPlot::PopStyleVar(4);
    ImPlot::EndPlot();
  }
}

void App::RenderThresholdOptimizationChart() {
  RenderChartHeader("THRESHOLD SENSITIVITY", "Analysis of how selection thresholds impact model precision vs. recall. The 'Sweet Spot' is highlighted where both metrics are balanced.");

  const auto& opt = loader_.GetOptimizationData();
  if (opt.threshold_sensitivity.empty()) {
    ImGui::TextDisabled("No threshold data available.");
    return;
  }

    if (ImPlot::BeginPlot("##Thresholds", ImGui::GetContentRegionAvail())) {
      ImPlot::SetupAxes("Threshold", "Sensitivity");
      ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1.0, ImPlotCond_Once);
      ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.0, ImPlotCond_Once);
      ApplyPremiumPlotStyles("##Thresholds");
      
      // Goal Region Shading Overlay
      double gx[2] = {0.6, 0.8}; 
      double gy1[2] = {0.0, 0.0};
      double gy2[2] = {1.0, 1.0};
      ImPlot::SetNextFillStyle(ImVec4(1, 0.8f, 0, 0.1f));
      ImPlot::PlotShaded("Sweet Spot", gx, gy1, gy2, 2);

      std::vector<float> xs, ys;
      for (const auto& [t, s] : opt.threshold_sensitivity) {
        xs.push_back(std::stof(t));
        ys.push_back(s);
      }
      
      if (!xs.empty()) {
        ImPlot::PlotLine("Sensitivity", xs.data(), ys.data(), xs.size());
      }
      
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(4);
      ImPlot::EndPlot();
    }
}

void App::RenderSidebar() {
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 2));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.04f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.08f));

  auto sidebar_button = [&](const char* label, Workspace ws, const char* icon) {
    bool active = current_workspace_ == ws;
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.85f, 1.0f, 1.0f));
    }

    ImGui::PushID(label);
    ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, 40);
    if (ImGui::Button("##hidden", size)) {
      current_workspace_ = ws;
      force_reset_layout_ = true;
    }
    
    // Custom drawing over the button
    ImVec2 p_min = ImGui::GetItemRectMin();
    ImVec2 p_max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    
    if (active) {
        draw->AddRectFilled(p_min, ImVec2(p_min.x + 3, p_max.y), ImColor(102, 217, 255));
        draw->AddRectFilled(p_min, p_max, ImColor(102, 217, 255, 10));
    }

    ImVec2 icon_pos = ImVec2(p_min.x + size.x * 0.5f - 10, p_min.y + 15);
    ImGui::SetCursorScreenPos(ImVec2(p_min.x + 15, p_min.y + 10));
    ImGui::BeginGroup();
    
    // PUSH FONT FOR ICON
    if (font_ui_) ImGui::PushFont(font_ui_);
    ImGui::Text("%s  %s", icon, label);
    if (font_ui_) ImGui::PopFont();
    
    ImGui::EndGroup();

    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s workspace", label);
    }
    ImGui::PopID();

    if (active) ImGui::PopStyleColor();
  };

  ImGui::Spacing();
  ImGui::Spacing();
  
  ImGui::PushFont(font_header_);
  ImGui::SetCursorPosX(20);
  ImGui::TextDisabled("WORKSPACES");
  ImGui::PopFont();
  ImGui::Spacing();

  sidebar_button("Dashboard", Workspace::Dashboard, ICON_MD_DASHBOARD);
  sidebar_button("Analysis", Workspace::Analysis, ICON_MD_ANALYTICS);
  sidebar_button("Optimization", Workspace::Optimization, ICON_MD_SETTINGS_INPUT_COMPONENT);
  sidebar_button("Systems", Workspace::Systems, ICON_MD_ROUTER);
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::PushFont(font_header_);
  ImGui::SetCursorPosX(20);
  ImGui::TextDisabled("COLLABORATION");
  ImGui::PopFont();
  ImGui::Spacing();

  sidebar_button("Chat", Workspace::Chat, ICON_MD_CHAT);
  sidebar_button("Training", Workspace::Training, ICON_MD_MODEL_TRAINING);
  sidebar_button("Context", Workspace::Context, ICON_MD_FOLDER_OPEN);

  ImGui::PopStyleColor(3);
  ImGui::PopStyleVar();
}

void App::RenderMetricCards() {
  const auto& generator_stats = loader_.GetGeneratorStats();
  const auto& trends = loader_.GetQualityTrends();

  float avg_acceptance = 0.0f;
  float total_success = 0.0f;
  for (const auto& agent : agents_) {
      total_success += agent.success_rate;
  }
  if (!agents_.empty())
      total_success /= static_cast<float>(agents_.size());

  float avg_quality = 0.0f;
  if (!trends.empty()) {
    for (const auto& t : trends)
      avg_quality += t.mean;
    avg_quality /= static_cast<float>(trends.size());
  }

  char q_buf[32], a_buf[32];
  snprintf(q_buf, sizeof(q_buf), "%d%%", static_cast<int>(avg_quality * 100));
  snprintf(a_buf, sizeof(a_buf), "%d%%", static_cast<int>(total_success * 100));

  MetricCard cards[] = {
      {ICON_MD_INSIGHTS "  Overall Quality", q_buf, "System Health",
       GetThemeColor(ImGuiCol_PlotLines), true},
      {ICON_MD_SPEED "  Swarm Velocity", "1.2k/s", "+12% vs last run",
       ImVec4(0.4f, 1.0f, 0.6f, 1.0f), true},
      {ICON_MD_AUTO_FIX_HIGH "  Efficiency", a_buf, "Mean Success Rate", ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
       total_success > 0.85f}};

  float card_w = (ImGui::GetContentRegionAvail().x - 16) / 3.0f;
  
  for (int i = 0; i < 3; ++i) {
    ImGui::PushID(i);
    ImGui::BeginChild("Card", ImVec2(card_w, 100), true, ImGuiWindowFlags_NoScrollbar);
    
    // Label
    if (font_ui_) ImGui::PushFont(font_ui_);
    ImGui::TextDisabled("%s", cards[i].label.c_str());
    if (font_ui_) ImGui::PopFont();
    
    // Value
    if (font_header_) ImGui::PushFont(font_header_);
    ImGui::TextColored(cards[i].color, "%s", cards[i].value.c_str());
    if (font_header_) ImGui::PopFont();
    
    // Subtext
    ImGui::Spacing();
    ImGui::TextDisabled("%s", cards[i].sub_text.c_str());
    
    // Decorative Pulse (Bottom edge)
    if (use_pulse_animations_) {
        float p = (1.0f + std::sin(pulse_timer_ * 2.0f + i)) * 0.5f;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        ImVec2 p_min = ImGui::GetItemRectMin();
        ImVec2 p_max = ImGui::GetWindowPos();
        p_max.x += ImGui::GetWindowSize().x;
        p_max.y += ImGui::GetWindowSize().y;
        draw->AddRectFilled(ImVec2(p_max.x - 40 * p, p_max.y - 2), p_max, ImColor(cards[i].color));
    }

    ImGui::EndChild();
    if (i < 2) ImGui::SameLine();
    ImGui::PopID();
  }
}

void App::ApplyPremiumPlotStyles(const char* plot_id) {
  // Custom theme-like styling for HAFS
  ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f); // More vibrant fill
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.8f);
  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 4.5f);
  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, 1.2f);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(12, 12));
  ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding, ImVec2(5, 5));
  
  // Standard grid line color (subtle)
  ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(1, 1, 1, 0.15f));
  ImPlot::PushStyleColor(ImPlotCol_Line, GetThemeColor(ImGuiCol_PlotLines));
}

void App::RenderDashboardView() {
  RenderSummaryRow();
  
  float avail_y = ImGui::GetContentRegionAvail().y;
  float hero_height = std::max(400.0f, avail_y * 0.6f);
  float card_height = std::max(200.0f, avail_y * 0.4f);

  if (ImGui::BeginTable("DashboardGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("QualityHero", ImVec2(0, hero_height), true)) {
      RenderQualityChart();
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("ThroughputHero", ImVec2(0, hero_height), true)) {
      RenderAgentThroughputChart();
    }
    ImGui::EndChild();
    
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("CoverageCard", ImVec2(0, card_height), true)) {
      RenderCoverageChart();
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("RejectionCard", ImVec2(0, card_height), true)) {
      RenderRejectionChart();
    }
    ImGui::EndChild();
    
    ImGui::EndTable();
  }
}

void App::RenderAnalysisView() {
  float avail_y = ImGui::GetContentRegionAvail().y;
  float main_height = std::max(600.0f, avail_y - 20.0f);

  if (ImGui::BeginTable("AnalysisGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("ConceptualGraph", ImGuiTableColumnFlags_WidthStretch, 0.5f);
    ImGui::TableSetupColumn("LatentSpace", ImGuiTableColumnFlags_WidthStretch, 0.5f);
    
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("KnowledgeCard", ImVec2(0, main_height), true)) {
      ImGui::TextDisabled("CONCEPTUAL KNOWLEDGE GRAPH");
      RenderKnowledgeGraph();
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("LatentCard", ImVec2(0, main_height), true)) {
      ImGui::TextDisabled("LATENT EMBEDDING SPACE (uMAP)");
      RenderLatentSpaceChart();
    }
    ImGui::EndChild();
    
    ImGui::EndTable();
  }
}

void App::RenderOptimizationView() {
  float card_height = 450.0f;
  
  if (ImGui::BeginChild("EffectivenessCard", ImVec2(0, card_height), true)) {
    RenderEffectivenessChart();
  }
  ImGui::EndChild();
  
  ImGui::Spacing();
  
  if (ImGui::BeginChild("OptimizationCard", ImVec2(0, card_height), true)) {
    RenderThresholdOptimizationChart();
  }
  ImGui::EndChild();
}

void App::RenderSystemsView() {
  float card_height = 400.0f;
  if (ImGui::BeginTable("SystemsGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("ControlsCard", ImVec2(0, card_height), true)) {
      RenderControlColumn();
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("GenStatsCard", ImVec2(0, card_height), true)) {
      RenderGeneratorChart();
    }
    ImGui::EndChild();
    
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("HistoryCard", ImVec2(0, card_height), true)) {
      RenderTrainingChart();
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("LogsCard", ImVec2(0, card_height), true)) {
      RenderLogsTab();
    }
    ImGui::EndChild();
    
    ImGui::EndTable();
  }
}


void App::RenderChatView() {
  RenderChartHeader(ICON_MD_CHAT " HAFS CHAT", "Interact with the HAFS agent swarm via natural language. Submit queries, get analysis, and issue commands.");
  
  ImGui::BeginChild("ChatHistory", ImVec2(0, -60), true);
    ImGui::TextWrapped("Welcome to HAFS Chat. Type a message below to interact with the agent swarm.");
    ImGui::Separator();
    ImGui::TextDisabled("[System] Session initialized. Ready for queries.");
  ImGui::EndChild();
  
  ImGui::Separator();
  ImGui::InputTextMultiline("##ChatInput", chat_input_.data(), chat_input_.size(), ImVec2(-1, 50));
  ImGui::SameLine();
  if (ImGui::Button("Send")) {
     // TODO: Implement send logic
  }
}

void App::RenderTrainingView() {
  RenderChartHeader(ICON_MD_MODEL_TRAINING " MODEL TRAINING", "Configure and monitor fine-tuning jobs for the HAFS agent models.");
  
  if (ImGui::BeginTable("TrainingGrid", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextRow();
    
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("TrainingConfig", ImVec2(0, 300), true)) {
      ImGui::Text(ICON_MD_SETTINGS " Training Configuration");
      ImGui::Separator();
      ImGui::InputText("Model Base", user_prompt_.data(), user_prompt_.size());
      ImGui::SliderFloat("Learning Rate", &quality_threshold_, 0.0001f, 0.1f, "%.5f");
      ImGui::SliderInt("Epochs", &mission_concurrency_, 1, 100);
      ImGui::Spacing();
      if (ImGui::Button(ICON_MD_PLAY_ARROW " Start Training")) {
        // TODO: Implement training logic
      }
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("TrainingProgress", ImVec2(0, 300), true)) {
      RenderTrainingLossChart();
    }
    ImGui::EndChild();
    
    ImGui::EndTable();
  }
}

void App::RenderContextView() {
  RenderChartHeader(ICON_MD_FOLDER_OPEN " AFS CONTEXT BROWSER", "Browse and manage files in your Agentic File System. Select files to add to agent context.");
  
  // AFS Only Filter Toggle
  static bool afs_only = true;
  if (ImGui::Checkbox(ICON_MD_FILTER_ALT " AFS Context Only", &afs_only)) {
      RefreshBrowserEntries(); // Refresh not strictly needed but good signal
  }
  ImGui::SameLine();
  ImGui::TextDisabled("(Shows Projects with .context and Global Context)");
  ImGui::Separator();
  
  float window_height = ImGui::GetContentRegionAvail().y - 10;
  
  if (ImGui::BeginTable("ContextBrowser", 2, ImGuiTableFlags_Resizable)) {
    ImGui::TableNextRow();
    
    ImGui::TableSetColumnIndex(0);
    if (ImGui::BeginChild("FileTree", ImVec2(0, window_height), true)) {
      // Breadcrumbs / Current Path
      ImGui::Text(ICON_MD_FOLDER " %s", current_browser_path_.c_str());
      ImGui::SameLine();
      if (ImGui::Button(ICON_MD_REFRESH)) {
          RefreshBrowserEntries();
      }
      ImGui::Separator();
      
      // Render file entries
      for (size_t i = 0; i < browser_entries_.size(); ++i) {
        const auto& entry = browser_entries_[i];
        
        // AFS Filtering Logic
        if (afs_only) {
            bool is_context_root = entry.name == ".context";
            bool is_parent = entry.name == "..";
            bool inside_context = current_browser_path_.string().find(".context") != std::string::npos;
            
            if (!is_parent && !inside_context && !is_context_root) {
                // If we are navigating normal folders, ONLY show those that contain a .context child
                // Check if directory contains .context
                if (entry.is_directory && entry.name != ".context") {
                     bool has_context = std::filesystem::exists(entry.path / ".context");
                     if (!has_context) continue; // Skip non-project folders
                } else if (!entry.is_directory) {
                    continue; // Skip files outside of context roots
                }
            }
        }

        bool is_dir = entry.is_directory;
        const char* icon = is_dir ? ICON_MD_FOLDER : ICON_MD_DESCRIPTION;
        if (entry.name == ".context") icon = ICON_MD_SETTINGS_SYSTEM_DAYDREAM;
        
        // Selection Logic
        bool selected = false; // TODO: Track selection
        if (ImGui::Selectable(std::string(std::string(icon) + " " + entry.name).c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
            if (ImGui::IsMouseDoubleClicked(0)) {
                if (is_dir) {
                    current_browser_path_ = entry.path;
                    RefreshBrowserEntries();
                } else {
                    LoadFile(entry.path);
                }
            }
        }
      }
    }
    ImGui::EndChild();
    
    ImGui::TableSetColumnIndex(1);
    if (ImGui::BeginChild("FilePreview", ImVec2(0, window_height), true)) {
      if (selected_file_path_.empty()) {
          ImGui::TextDisabled("Select a file to preview its contents.");
      } else {
          ImGui::Text("%s %s", ICON_MD_DESCRIPTION, selected_file_path_.filename().string().c_str());
          ImGui::SameLine();
          ImGui::TextDisabled("(%s)", is_binary_view_ ? "Binary/Hex View" : "Text Editor");
          ImGui::Separator();
          
          if (is_binary_view_) {
              memory_editor_.DrawContents(binary_data_.data(), binary_data_.size());
          } else {
              text_editor_.Render("TextEditor", ImVec2(0,0), false);
          }
      }
    }
    ImGui::EndChild();
    
    ImGui::EndTable();
  }
}

ImVec4 App::GetThemeColor(ImGuiCol col) {
  switch (current_theme_) {
    case ThemeProfile::Amber:
      if (col == ImGuiCol_Text) return ImVec4(1.0f, 0.7f, 0.0f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(1.0f, 0.6f, 0.0f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
      break;
    case ThemeProfile::Emerald:
      if (col == ImGuiCol_Text) return ImVec4(0.0f, 1.0f, 0.4f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.0f, 0.8f, 0.3f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.2f, 1.0f, 0.5f, 1.0f);
      break;
    default: // Cobalt
      if (col == ImGuiCol_PlotLines) return ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
      break;
  }
  return ImGui::GetStyleColorVec4(col);
}

ImVec4 App::GetStepColor(float step) {
  ImVec4 base = GetThemeColor(ImGuiCol_PlotLines);
  float alpha = 0.2f + 0.8f * step;
  return ImVec4(base.x, base.y, base.z, alpha);
}

void App::HelpMarker(const char* desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

void App::RenderChartHeader(const char* title, const char* desc) {
  ImGui::PushStyleColor(ImGuiCol_Text, GetThemeColor(ImGuiCol_PlotLines));
  ImGui::Text("%s", title);
  ImGui::PopStyleColor();
  ImGui::SameLine();
  HelpMarker(desc);
  ImGui::Spacing();
}

void App::RefreshBrowserEntries() {
  browser_entries_.clear();
  
  // Add parent directory ".." if not at root
  if (current_browser_path_.has_parent_path() && current_browser_path_ != current_browser_path_.root_path()) {
      browser_entries_.push_back({"..", current_browser_path_.parent_path(), true, 0});
  }

  // AFS Only Logic:
  // If true, we only show:
  // 1. Directories that contain a .context folder (Projects)
  // 2. The .context folder itself (Global or Project-Local)
  // 3. Files/Dirs if we are INSIDE a .context folder
  
  // Check if we are inside a .context folder
  bool inside_context = current_browser_path_.string().find(".context") != std::string::npos;
  
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(current_browser_path_, ec)) {
    if (entry.is_directory()) {
        browser_entries_.push_back({
            entry.path().filename().string(),
            entry.path(),
            true,
            0
        });
    } else if (entry.is_regular_file()) {
        browser_entries_.push_back({
            entry.path().filename().string(),
            entry.path(),
            false,
            entry.file_size()
        });
    }
  }
  
  // Sort: Directories first, then alphabetical
  std::sort(browser_entries_.begin(), browser_entries_.end(), [](const FileEntry& a, const FileEntry& b) {
      if (a.is_directory != b.is_directory) return a.is_directory > b.is_directory;
      return a.name < b.name;
  });
}

void App::LoadFile(const std::filesystem::path& path) {
  selected_file_path_ = path;
  std::string ext = path.extension().string();
  
  // Simple heuristic for binary vs text
  // Extensions known to be text
  static const std::vector<std::string> text_exts = {
      ".cpp", ".cc", ".c", ".h", ".hpp", ".py", ".md", ".json", ".txt", ".xml", ".org", ".asm", ".s", ".cmake", ".yml", ".yaml", ".sh"
  };
  
  bool is_text = false;
  for (const auto& e : text_exts) {
      if (ext == e) {
          is_text = true;
          break;
      }
  }
  
  if (is_text) {
      is_binary_view_ = false;
      std::ifstream t(path);
      if (t.is_open()) {
          std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
          text_editor_.SetText(str);
          
          // Set Language Definition based on extension
          if (ext == ".cpp" || ext == ".cc" || ext == ".h" || ext == ".hpp") {
             text_editor_.SetLanguageDefinition(TextEditor::LanguageDefinition::CPlusPlus());
          } else if (ext == ".py") {
             // TODO: Python def
             text_editor_.SetLanguageDefinition(TextEditor::LanguageDefinition::CPlusPlus()); // Placeholder
          } else if (ext == ".sql") {
             text_editor_.SetLanguageDefinition(TextEditor::LanguageDefinition::SQL());
          } else {
             text_editor_.SetLanguageDefinition(TextEditor::LanguageDefinition()); // Plain text
          }
      }
  } else {
      // Default to binary (Memory Editor) or maybe simple text check?
      // User asked for "default file viewer -> unknown filetypes". 
      // yaze TextEditor handles plain text well.
      // Let's try to read it as binary first.
      is_binary_view_ = true;
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (file.is_open()) {
          std::streamsize size = file.tellg();
          file.seekg(0, std::ios::beg);
          
          if (size > 10 * 1024 * 1024) { // Limit to 10MB
              // Too large
              binary_data_.clear(); 
          } else {
              binary_data_.resize(size);
              if (file.read((char*)binary_data_.data(), size)) {
                  // Success
              }
          }
      }
  }
}

}  // namespace viz
}  // namespace hafs
