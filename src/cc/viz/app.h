#pragma once

#include <memory>
#include <string>

#include "data_loader.h"

// Forward declarations - implot.h and imgui.h included in .cc
struct ImGuiContext;
struct ImPlotContext;
struct GLFWwindow;

namespace hafs {
namespace viz {

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

  void RenderFrame();
  void RenderMenuBar();
  void RenderDashboard();
  void RenderQualityChart();
  void RenderGeneratorChart();
  void RenderCoverageChart();
  void RenderTrainingChart();
  void RenderRejectionChart();
  void RenderStatusBar();

  std::string data_path_;
  DataLoader loader_;

  GLFWwindow* window_ = nullptr;
  ImGuiContext* imgui_ctx_ = nullptr;
  ImPlotContext* implot_ctx_ = nullptr;

  int window_width_ = 1400;
  int window_height_ = 900;
  bool should_refresh_ = false;
  bool show_demo_window_ = false;
};

}  // namespace viz
}  // namespace hafs
