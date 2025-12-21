#include "app.h"

#include <algorithm>
#include <cstdio>
#include <vector>

// GLFW + OpenGL
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// ImPlot
#include "implot.h"

namespace hafs {
namespace viz {

namespace {

void GlfwErrorCallback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

}  // namespace

App::App(const std::string& data_path)
    : data_path_(data_path), loader_(data_path) {}

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

  // Setup style - dark theme with custom colors
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 4.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;

  // Custom colors matching HAFS theme
  ImVec4* colors = style.Colors;
  colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.30f, 0.40f, 1.00f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.40f, 0.50f, 1.00f);
  colors[ImGuiCol_Button] = ImVec4(0.15f, 0.35f, 0.55f, 1.00f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.45f, 0.70f, 1.00f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.50f, 0.80f, 1.00f);
  colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.30f, 0.85f, 0.70f, 1.00f);

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

int App::Run() {
  if (!InitWindow()) return 1;
  if (!InitImGui()) return 1;

  // Initial data load
  loader_.Refresh();

  // Main loop
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    // Handle refresh request
    if (should_refresh_) {
      loader_.Refresh();
      should_refresh_ = false;
    }

    // Handle F5 for refresh
    if (glfwGetKey(window_, GLFW_KEY_F5) == GLFW_PRESS) {
      should_refresh_ = true;
    }

    RenderFrame();
  }

  return 0;
}

void App::RenderFrame() {
  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // Get window size for full-screen layout
  glfwGetFramebufferSize(window_, &window_width_, &window_height_);

  RenderMenuBar();
  RenderDashboard();
  RenderStatusBar();

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
      if (ImGui::MenuItem("Show Demo Windows", nullptr, &show_demo_window_)) {}
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

void App::RenderDashboard() {
  // Use available space below menu bar
  float menu_height = ImGui::GetFrameHeight();
  float status_height = ImGui::GetFrameHeight() + 8;

  ImGui::SetNextWindowPos(ImVec2(0, menu_height));
  ImGui::SetNextWindowSize(
      ImVec2(static_cast<float>(window_width_),
             static_cast<float>(window_height_) - menu_height - status_height));

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoCollapse;

  ImGui::Begin("Dashboard", nullptr, flags);

  if (!loader_.HasData()) {
    ImGui::TextWrapped("No training data found in: %s", data_path_.c_str());
    ImGui::TextWrapped("Expected files: quality_feedback.json, "
                       "active_learning.json, training_feedback.json");
  } else {
    // 2x2 grid layout using tables
    if (ImGui::BeginTable("ChartGrid", 2,
                          ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_BordersInner)) {
      float row_height =
          (static_cast<float>(window_height_) - menu_height - status_height -
           60) /
          2.0f;

      // Row 1
      ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);

      ImGui::TableNextColumn();
      ImGui::BeginChild("QualityPanel", ImVec2(0, row_height - 20), true);
      ImGui::Text("Quality Trends");
      ImGui::Separator();
      RenderQualityChart();
      ImGui::EndChild();

      ImGui::TableNextColumn();
      ImGui::BeginChild("GeneratorPanel", ImVec2(0, row_height - 20), true);
      ImGui::Text("Generator Stats");
      ImGui::Separator();
      RenderGeneratorChart();
      ImGui::EndChild();

      // Row 2
      ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);

      ImGui::TableNextColumn();
      ImGui::BeginChild("CoveragePanel", ImVec2(0, row_height - 20), true);
      ImGui::Text("Embedding Coverage");
      ImGui::Separator();
      RenderCoverageChart();
      ImGui::EndChild();

      ImGui::TableNextColumn();
      ImGui::BeginChild("TrainingPanel", ImVec2(0, row_height - 20), true);
      ImGui::Text("Training Runs");
      ImGui::Separator();
      RenderTrainingChart();
      ImGui::EndChild();

      ImGui::EndTable();
    }
  }

  ImGui::End();
}

void App::RenderQualityChart() {
  const auto& trends = loader_.GetQualityTrends();
  if (trends.empty()) {
    ImGui::TextDisabled("No quality trend data available");
    return;
  }

  if (ImPlot::BeginPlot("##QualityTrends", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Sample Index", "Score");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);

    for (size_t i = 0; i < std::min(trends.size(), size_t(5)); ++i) {
      const auto& trend = trends[i];
      if (!trend.values.empty()) {
        std::string label = trend.domain + "/" + trend.metric;

        // Create x-axis indices
        std::vector<float> xs(trend.values.size());
        for (size_t j = 0; j < xs.size(); ++j) {
          xs[j] = static_cast<float>(j);
        }

        ImPlot::PlotLine(label.c_str(), xs.data(), trend.values.data(),
                         static_cast<int>(trend.values.size()));
      }
    }

    ImPlot::EndPlot();
  }
}

void App::RenderGeneratorChart() {
  const auto& stats = loader_.GetGeneratorStats();
  if (stats.empty()) {
    ImGui::TextDisabled("No generator stats available");
    return;
  }

  // Prepare data for bar chart
  std::vector<const char*> labels;
  std::vector<float> rates;
  std::vector<std::string> label_storage;  // Keep strings alive

  for (const auto& s : stats) {
    // Remove "DataGenerator" suffix for cleaner labels
    std::string name = s.name;
    size_t pos = name.find("DataGenerator");
    if (pos != std::string::npos) {
      name = name.substr(0, pos);
    }
    label_storage.push_back(name);
    rates.push_back(s.acceptance_rate * 100.0f);
  }

  for (const auto& s : label_storage) {
    labels.push_back(s.c_str());
  }

  if (ImPlot::BeginPlot("##GeneratorStats", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Generator", "Acceptance %");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());

    ImPlot::PlotBars("Rate", rates.data(), static_cast<int>(rates.size()), 0.67);

    ImPlot::EndPlot();
  }
}

void App::RenderCoverageChart() {
  const auto& regions = loader_.GetEmbeddingRegions();
  const auto& coverage = loader_.GetCoverage();

  if (regions.empty()) {
    ImGui::TextDisabled("No embedding coverage data available");
    return;
  }

  ImGui::Text("Coverage Score: %.1f%%", coverage.coverage_score * 100.0f);
  ImGui::SameLine();
  ImGui::Text("  Sparse Regions: %d", coverage.sparse_regions);

  // Scatter plot of region densities
  std::vector<float> xs, dense_x, dense_y, sparse_x, sparse_y;

  // Calculate average for sparse/dense classification
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

  if (ImPlot::BeginPlot("##Coverage", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Region Index", "Sample Count");

    if (!dense_x.empty()) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, ImVec4(0, 0.8f, 0.4f, 1));
      ImPlot::PlotScatter("Dense", dense_x.data(), dense_y.data(),
                          static_cast<int>(dense_x.size()));
    }

    if (!sparse_x.empty()) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 6, ImVec4(1, 0.3f, 0.3f, 1));
      ImPlot::PlotScatter("Sparse", sparse_x.data(), sparse_y.data(),
                          static_cast<int>(sparse_x.size()));
    }

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

  if (ImPlot::BeginPlot("##TrainingLoss", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Run", "Final Loss");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());

    ImPlot::PlotBars("Loss", losses.data(), static_cast<int>(losses.size()), 0.67);

    ImPlot::EndPlot();
  }
}

void App::RenderRejectionChart() {
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

  if (ImPlot::BeginPlot("##Rejections", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Reason", "Count");
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                           static_cast<int>(labels.size()), labels.data());

    ImPlot::PlotBars("Count", counts.data(), static_cast<int>(counts.size()),
                     0.67);

    ImPlot::EndPlot();
  }
}

void App::RenderStatusBar() {
  ImGuiViewport* viewport = ImGui::GetMainViewport();
  float status_height = ImGui::GetFrameHeight() + 8;

  ImGui::SetNextWindowPos(
      ImVec2(viewport->WorkPos.x,
             viewport->WorkPos.y + viewport->WorkSize.y - status_height));
  ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_height));

  ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBringToFrontOnFocus;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
  ImGui::Begin("StatusBar", nullptr, flags);

  if (loader_.HasData()) {
    ImGui::Text("Generators: %zu  |  Regions: %zu  |  Runs: %zu  |  F5 to refresh",
                loader_.GetGeneratorStats().size(),
                loader_.GetEmbeddingRegions().size(),
                loader_.GetTrainingRuns().size());
  } else {
    ImGui::TextDisabled("No data loaded - Press F5 to refresh");
  }

  ImGui::SameLine(ImGui::GetWindowWidth() - 200);
  ImGui::Text("Data: %s", data_path_.c_str());

  ImGui::End();
  ImGui::PopStyleColor();
}

}  // namespace viz
}  // namespace hafs
