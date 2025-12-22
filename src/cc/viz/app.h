#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <functional>

#include <imgui.h>
#include "data_loader.h"
#include "models/state.h"
#include "ui/shortcuts.h"
#include "widgets/text_editor.h"
#include "widgets/imgui_memory_editor.h"
#include "widgets/sample_review.h"

// Forward declarations
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

  // Core Logic
  void RefreshData(const char* reason);
  void MaybeAutoRefresh();
  void SeedDefaultState();
  void SyncDataBackedState();
  void TickSimulatedMetrics(float dt);
  void HandleShortcuts();
  
  // Rendering
  void RenderFrame();
  void RenderLayout();
  
  // Workspace Views (Orchestration)
  void RenderDashboardView();
  void RenderAnalysisView();
  void RenderOptimizationView();
  void RenderSystemsView();
  void RenderCustomGridView();
  void RenderChatView();
  void RenderTrainingView();
  void RenderContextView();
  void RenderExpandedPlot();

  // Members
  std::string data_path_;
  DataLoader loader_;
  AppState state_;

  GLFWwindow* window_ = nullptr;
  ImGuiContext* imgui_ctx_ = nullptr;
  ImPlotContext* implot_ctx_ = nullptr;

  int window_width_ = 1400;
  int window_height_ = 900;

  // Editors
  TextEditor text_editor_;
  MemoryEditorWidget memory_editor_;
  SampleReviewWidget sample_review_;
  bool show_sample_review_ = false;
  ui::ShortcutManager shortcut_manager_;
  bool show_shortcuts_window_ = false;

  // Typography
  ImFont* font_ui_ = nullptr;
  ImFont* font_header_ = nullptr;
  ImFont* font_mono_ = nullptr;
  ImFont* font_icons_ = nullptr;
};

}  // namespace viz
}  // namespace hafs
