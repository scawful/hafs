#include "panels.h"
#include "../core.h"
#include "../../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <GLFW/glfw3.h>

namespace hafs {
namespace viz {
namespace ui {

static bool ContainsInsensitive(const std::string& str, const std::string& pattern) {
  if (pattern.empty()) return true;
  auto it = std::search(str.begin(), str.end(), pattern.begin(), pattern.end(),
                        [](char ch1, char ch2) {
                          return std::tolower(static_cast<unsigned char>(ch1)) ==
                                 std::tolower(static_cast<unsigned char>(ch2));
                        });
  return it != str.end();
}

void RenderInspectorPanel(AppState& state, const DataLoader& loader, ImFont* font_header, const std::string& data_path) {
  const auto& trends = loader.GetQualityTrends();
  const auto& coverage = loader.GetCoverage();
  const auto& runs = loader.GetTrainingRuns();
  const auto& generators = loader.GetGeneratorStats();

  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_INSIGHTS " INSPECTOR");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  if (state.inspector_context != PlotKind::None) {
      ImGui::PushStyleColor(ImGuiCol_Text, GetThemeColor(ImGuiCol_PlotLines, state.current_theme));
      ImGui::Text(ICON_MD_SETTINGS " CHART PROPERTIES");
      ImGui::PopStyleColor();
      
      const auto& options = PlotOptions();
      const char* label = "Unknown Plot";
      for (const auto& opt : options) {
          if (opt.kind == state.inspector_context) { label = opt.label; break; }
      }
      ImGui::Text("Subject: %s", label);
      ImGui::Separator();
      
      ImGui::Checkbox("Show Markers", &state.show_markers);
      ImGui::Checkbox("Show Legend", &state.show_legend);
      ImGui::SliderFloat("Line Weight", &state.line_weight, 1.0f, 5.0f);
      
      if (ImGui::Button("Reset Context")) state.inspector_context = PlotKind::None;
      ImGui::Separator();
  }

  ImGui::TextDisabled("Data Snapshot");
  ImGui::Text("Runs: %zu", runs.size());
  ImGui::Text("Generators: %zu", generators.size());
  ImGui::Text("Regions: %zu", loader.GetEmbeddingRegions().size());
  ImGui::Text("Sparse Regions: %d", coverage.sparse_regions);
  ImGui::Text("Data Path:");
  ImGui::TextWrapped("%s", data_path.c_str());
  const auto& status = loader.GetLastStatus();
  ImGui::Text("Sources: %d/%d ok", status.OkCount(), status.FoundCount());
  if (status.error_count > 0) {
    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f),
                       "Load Errors: %d", status.error_count);
    if (!status.last_error.empty()) {
      ImGui::TextDisabled("Last Error:");
      ImGui::TextWrapped("%s (%s)",
                         status.last_error.c_str(),
                         status.last_error_source.empty()
                             ? "unknown"
                             : status.last_error_source.c_str());
    }
  }

  float avg_quality = 0.0f;
  if (!trends.empty()) {
    for (const auto& t : trends) avg_quality += t.mean;
    avg_quality /= static_cast<float>(trends.size());
  }

  ImGui::Spacing();
  ImGui::TextDisabled("Health Signals");
  ImGui::ProgressBar(avg_quality, ImVec2(-1, 0), "Avg Quality");
  ImGui::ProgressBar(coverage.coverage_score, ImVec2(-1, 0), "Coverage Score");

  ImGui::Separator();
  ImGui::TextDisabled("Selected Run");

  if (state.selected_run_index >= 0 && state.selected_run_index < static_cast<int>(runs.size())) {
    const auto& run = runs[state.selected_run_index];
    ImGui::Text("%s", run.run_id.c_str());
    if (!run.model_name.empty()) ImGui::Text("Model: %s", run.model_name.c_str());
    if (!run.base_model.empty()) ImGui::Text("Base: %s", run.base_model.c_str());
    ImGui::Text("Samples: %d", run.samples_count);
    ImGui::Text("Final Loss: %.5f", run.final_loss);
    if (!run.start_time.empty() || !run.end_time.empty()) {
      ImGui::Text("Window: %s -> %s", run.start_time.empty() ? "?" : run.start_time.c_str(), run.end_time.empty() ? "?" : run.end_time.c_str());
    }
    if (!run.dataset_path.empty()) {
      ImGui::Text("Dataset:");
      ImGui::TextWrapped("%s", run.dataset_path.c_str());
    }
    if (!run.notes.empty()) {
      ImGui::Text("Notes:");
      ImGui::TextWrapped("%s", run.notes.c_str());
    }

    if (!run.domain_distribution.empty()) {
      std::vector<std::pair<std::string, int>> domains(run.domain_distribution.begin(), run.domain_distribution.end());
      std::sort(domains.begin(), domains.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      if (domains.size() > 6) domains.resize(6);

      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [domain, count] : domains) {
        label_storage.push_back(domain);
        values.push_back(static_cast<float>(count));
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##RunDomains", state);
      if (ImPlot::BeginPlot("##RunDomains", ImVec2(-1, 140), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Domain", "Samples", axis_flags, axis_flags);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(2), 0.75f);
        ImPlot::PlotBars("Samples", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }

    if (!run.eval_metrics.empty()) {
      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [metric, value] : run.eval_metrics) {
        label_storage.push_back(metric);
        values.push_back(value);
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##RunMetrics", state);
      if (ImPlot::BeginPlot("##RunMetrics", ImVec2(-1, 120), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Metric", "Score", axis_flags, axis_flags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(4), 0.75f);
        ImPlot::PlotBars("Score", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }
  } else {
    ImGui::TextDisabled("Select a training run in the Dataset panel.");
  }

  ImGui::Separator();
  ImGui::TextDisabled("Selected Generator");
  if (state.selected_generator_index >= 0 && state.selected_generator_index < static_cast<int>(generators.size())) {
    const auto& gen = generators[state.selected_generator_index];
    ImGui::Text("%s", gen.name.c_str());
    ImGui::Text("Accepted: %d", gen.samples_accepted);
    ImGui::Text("Rejected: %d", gen.samples_rejected);
    ImGui::Text("Avg Quality: %.3f", gen.avg_quality);
    ImGui::ProgressBar(gen.acceptance_rate, ImVec2(-1, 0), "Acceptance Rate");

    if (!gen.rejection_reasons.empty()) {
      std::vector<std::pair<std::string, int>> reasons(gen.rejection_reasons.begin(), gen.rejection_reasons.end());
      std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      if (reasons.size() > 6) reasons.resize(6);

      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [reason, count] : reasons) {
        std::string formatted = reason;
        std::replace(formatted.begin(), formatted.end(), '_', ' ');
        label_storage.push_back(formatted);
        values.push_back(static_cast<float>(count));
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##GenRejections", state);
      if (ImPlot::BeginPlot("##GenRejections", ImVec2(-1, 120), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Reason", "Count", axis_flags, axis_flags);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(7), 0.75f);
        ImPlot::PlotBars("Count", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }
  } else {
    ImGui::TextDisabled("Select a generator in the Dataset panel.");
  }

  ImGui::Separator();
  ImGui::Spacing();
  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_FILE_DOWNLOAD_DONE " DATA INTEGRITY AUDIT");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  auto render_audit_item = [](const char* name, bool found, bool ok, const char* error) {
      ImGui::BeginGroup();
      if (!found) {
          ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_MD_FILE_OPEN " %s", name);
          ImGui::SameLine(); ImGui::TextDisabled("(Not Found)");
      } else if (!ok) {
          ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), ICON_MD_REPORT_GMAILERRORRED " %s", name);
          ImGui::SameLine(); ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "(Error)");
          if (error && error[0] != '\0') {
              ImGui::Indent();
              ImGui::TextDisabled("%s", error);
              ImGui::Unindent();
          }
      } else {
          ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ICON_MD_CHECK_CIRCLE " %s", name);
          ImGui::SameLine(); ImGui::TextDisabled("(Synced)");
      }
      ImGui::EndGroup();
  };

  render_audit_item("quality_feedback.json", status.quality_found, status.quality_ok, status.quality_ok ? "" : status.last_error.c_str());
  render_audit_item("active_learning.json", status.active_found, status.active_ok, status.active_ok ? "" : status.last_error.c_str());
  render_audit_item("training_feedback.json", status.training_found, status.training_ok, status.training_ok ? "" : status.last_error.c_str());

  ImGui::Spacing();
  ImGui::TextDisabled("Integrity Score: %.1f%%", (status.OkCount() / (float)std::max(1, status.FoundCount())) * 100.0f);
  ImGui::ProgressBar(status.OkCount() / (float)std::max(1, status.FoundCount()), ImVec2(-1, 0));
}

void RenderDatasetPanel(AppState& state, const DataLoader& loader) {
  const auto& runs = loader.GetTrainingRuns();
  const auto& generators = loader.GetGeneratorStats();
  const auto& coverage = loader.GetCoverage();

  if (ImGui::BeginTabBar("DatasetTabs")) {
    if (ImGui::BeginTabItem("Training Runs")) {
      ImGui::InputTextWithHint("##RunFilter", "Filter by run ID or model", state.run_filter.data(), state.run_filter.size());
      ImGui::SameLine();
      if (ImGui::Button("Clear")) state.run_filter[0] = '\0';

      if (ImGui::BeginTable("RunTable", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Run ID", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Samples", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Loss", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Domains", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < runs.size(); ++i) {
          const auto& run = runs[i];
          if (!ContainsInsensitive(run.run_id, state.run_filter.data()) && !ContainsInsensitive(run.model_name, state.run_filter.data())) continue;

          bool selected = static_cast<int>(i) == state.selected_run_index;
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          if (ImGui::Selectable(run.run_id.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_run_index = static_cast<int>(i);
            state.selected_run_id = run.run_id;
          }
          ImGui::TableNextColumn(); ImGui::Text("%s", run.model_name.empty() ? "-" : run.model_name.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%d", run.samples_count);
          ImGui::TableNextColumn(); ImGui::Text("%.4f", run.final_loss);
          ImGui::TableNextColumn(); ImGui::Text("%zu", run.domain_distribution.size());
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Generators")) {
      ImGui::InputTextWithHint("##GenFilter", "Filter by generator name", state.generator_filter.data(), state.generator_filter.size());
      ImGui::SameLine();
      if (ImGui::Button("Clear##Gen")) state.generator_filter[0] = '\0';

      if (ImGui::BeginTable("GeneratorTable", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Generator", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Accepted", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Rejected", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Rate %", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Quality", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < generators.size(); ++i) {
          const auto& gen = generators[i];
          if (!ContainsInsensitive(gen.name, state.generator_filter.data())) continue;
          bool selected = static_cast<int>(i) == state.selected_generator_index;
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          if (ImGui::Selectable(gen.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_generator_index = static_cast<int>(i);
            state.selected_generator_name = gen.name;
          }
          ImGui::TableNextColumn(); ImGui::Text("%d", gen.samples_accepted);
          ImGui::TableNextColumn(); ImGui::Text("%d", gen.samples_rejected);
          ImGui::TableNextColumn(); ImGui::Text("%.1f", gen.acceptance_rate * 100.0f);
          ImGui::TableNextColumn(); ImGui::Text("%.3f", gen.avg_quality);
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Domains")) {
      if (coverage.domain_coverage.empty()) {
        ImGui::TextDisabled("No domain coverage data available.");
      } else {
        if (ImGui::BeginTable("DomainCoverageTable", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders)) {
          ImGui::TableSetupColumn("Domain", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Coverage %", ImGuiTableColumnFlags_WidthFixed, 110);
          ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableHeadersRow();

          for (const auto& [domain, value] : coverage.domain_coverage) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s", domain.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%.1f", value * 100.0f);
            ImGui::TableNextColumn(); ImGui::ProgressBar(value, ImVec2(-1, 0));
          }
          ImGui::EndTable();
        }
      }
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void RenderSystemsPanel(AppState& state, ImFont* font_header, std::function<void(const char*)> refresh_callback) {
  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_ROUTER " SYSTEMS OVERVIEW");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  if (ImGui::Button(ICON_MD_REFRESH " Refresh Now", ImVec2(-1, 0))) {
    if (refresh_callback) refresh_callback("ui");
  }
  ImGui::Spacing();

  double seconds_since = std::max(0.0, glfwGetTime() - state.last_refresh_time);
  ImGui::Text("Auto Refresh: %s", state.auto_refresh ? "On" : "Off");
  ImGui::Text("Interval: %.1fs", state.refresh_interval_sec);
  ImGui::Text("Last Refresh: %.0fs ago", seconds_since);
  ImGui::Text("Simulation: %s", state.simulate_activity ? "On" : "Off");
  ImGui::Text("Quality Threshold: %.2f", state.quality_threshold);
  ImGui::Text("Mission Concurrency: %d", state.mission_concurrency);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextDisabled("Swarm Snapshot");

  int active_count = 0;
  int total_queue = 0;
  float total_success = 0.0f;
  for (const auto& agent : state.agents) {
    if (agent.enabled) active_count++;
    total_queue += agent.queue_depth;
    total_success += agent.success_rate;
  }
  if (!state.agents.empty()) total_success /= static_cast<float>(state.agents.size());

  if (ImGui::BeginTable("SystemSnapshot", 2, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableNextColumn();
    ImGui::Text("Active Agents");
    ImGui::Text("Queue Depth");
    ImGui::Text("Avg Success");
    ImGui::TableNextColumn();
    ImGui::Text("%d / %d", active_count, static_cast<int>(state.agents.size()));
    ImGui::Text("%d", total_queue);
    ImGui::Text("%.1f%%", total_success * 100.0f);
    ImGui::EndTable();
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextDisabled("Core Services");

  auto render_row = [](const char* name, const char* status) {
    ImVec4 status_color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);
    if (strcmp(status, "Warning") == 0) status_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
    else if (strcmp(status, "Error") == 0) status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);

    ImGui::Bullet(); ImGui::SameLine();
    ImGui::Text("%s", name); ImGui::SameLine();
    ImGui::TextColored(status_color, "%s", status);
  };

  render_row("Orchestrator", "Active");
  render_row("Knowledge Base", "Active");
  render_row("Trainer", "Active");
  render_row("Embedding SVC", "Warning");
}

void RenderMenuBar(AppState& state,
                   std::function<void(const char*)> refresh_callback,
                   std::function<void()> quit_callback,
                   ShortcutManager& shortcuts,
                   bool* show_sample_review,
                   bool* show_shortcuts_window) {
  if (ImGui::BeginMainMenuBar()) {
    auto shortcut_label = [&](ActionId action_id) -> std::string {
      return shortcuts.FormatShortcut(action_id, ImGui::GetIO());
    };

    if (ImGui::BeginMenu("File")) {
      std::string refresh_shortcut = shortcut_label(ActionId::Refresh);
      if (ImGui::MenuItem("Refresh", refresh_shortcut.empty() ? nullptr
                                                              : refresh_shortcut.c_str())) {
        if (refresh_callback) refresh_callback("manual");
      }
      ImGui::Separator();
      std::string quit_shortcut = shortcut_label(ActionId::Quit);
      if (ImGui::MenuItem("Quit", quit_shortcut.empty() ? nullptr
                                                        : quit_shortcut.c_str())) {
        if (quit_callback) quit_callback();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::BeginMenu("Workspace")) {
        std::string dash_shortcut = shortcut_label(ActionId::WorkspaceDashboard);
        if (ImGui::MenuItem("Dashboard", dash_shortcut.empty() ? nullptr
                                                               : dash_shortcut.c_str(),
                            state.current_workspace == Workspace::Dashboard)) {
          state.current_workspace = Workspace::Dashboard;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string analysis_shortcut = shortcut_label(ActionId::WorkspaceAnalysis);
        if (ImGui::MenuItem("Analysis", analysis_shortcut.empty() ? nullptr
                                                                  : analysis_shortcut.c_str(),
                            state.current_workspace == Workspace::Analysis)) {
          state.current_workspace = Workspace::Analysis;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string optimization_shortcut = shortcut_label(ActionId::WorkspaceOptimization);
        if (ImGui::MenuItem("Optimization",
                            optimization_shortcut.empty() ? nullptr
                                                          : optimization_shortcut.c_str(),
                            state.current_workspace == Workspace::Optimization)) {
          state.current_workspace = Workspace::Optimization;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string systems_shortcut = shortcut_label(ActionId::WorkspaceSystems);
        if (ImGui::MenuItem("Systems", systems_shortcut.empty() ? nullptr
                                                                : systems_shortcut.c_str(),
                            state.current_workspace == Workspace::Systems)) {
          state.current_workspace = Workspace::Systems;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string custom_shortcut = shortcut_label(ActionId::WorkspaceCustom);
        if (ImGui::MenuItem("Custom Grid", custom_shortcut.empty() ? nullptr
                                                                   : custom_shortcut.c_str(),
                            state.current_workspace == Workspace::Custom)) {
          state.current_workspace = Workspace::Custom;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        ImGui::Separator();
        std::string chat_shortcut = shortcut_label(ActionId::WorkspaceChat);
        if (ImGui::MenuItem("Chat", chat_shortcut.empty() ? nullptr
                                                          : chat_shortcut.c_str(),
                            state.current_workspace == Workspace::Chat)) {
            state.current_workspace = Workspace::Chat;
            if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string training_shortcut = shortcut_label(ActionId::WorkspaceTraining);
        if (ImGui::MenuItem("Training Hub", training_shortcut.empty() ? nullptr
                                                                      : training_shortcut.c_str(),
                            state.current_workspace == Workspace::Training)) {
            state.current_workspace = Workspace::Training;
            if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string context_shortcut = shortcut_label(ActionId::WorkspaceContext);
        if (ImGui::MenuItem("Context Broker", context_shortcut.empty() ? nullptr
                                                                       : context_shortcut.c_str(),
                            state.current_workspace == Workspace::Context)) {
            state.current_workspace = Workspace::Context;
            if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        ImGui::EndMenu();
      }
      
      if (ImGui::BeginMenu("Theme Profile")) {
          if (ImGui::MenuItem("Deep Sea (Default)", nullptr, state.current_theme == ThemeProfile::Default)) state.current_theme = ThemeProfile::Default;
          if (ImGui::MenuItem("Cyberpunk", nullptr, state.current_theme == ThemeProfile::Cyberpunk)) state.current_theme = ThemeProfile::Cyberpunk;
          if (ImGui::MenuItem("Monochrome", nullptr, state.current_theme == ThemeProfile::Monochrome)) state.current_theme = ThemeProfile::Monochrome;
          if (ImGui::MenuItem("Solarized", nullptr, state.current_theme == ThemeProfile::Solarized)) state.current_theme = ThemeProfile::Solarized;
          if (ImGui::MenuItem("Nord", nullptr, state.current_theme == ThemeProfile::Nord)) state.current_theme = ThemeProfile::Nord;
          if (ImGui::MenuItem("Dracula", nullptr, state.current_theme == ThemeProfile::Dracula)) state.current_theme = ThemeProfile::Dracula;
          ImGui::EndMenu();
      }

      ImGui::Separator();
      if (ImGui::BeginMenu("Panels")) {
        std::string inspector_shortcut = shortcut_label(ActionId::ToggleInspector);
        ImGui::MenuItem("Inspector", inspector_shortcut.empty() ? nullptr
                                                                : inspector_shortcut.c_str(),
                        &state.show_inspector);
        std::string dataset_shortcut = shortcut_label(ActionId::ToggleDatasetPanel);
        ImGui::MenuItem("Dataset Panel", dataset_shortcut.empty() ? nullptr
                                                                  : dataset_shortcut.c_str(),
                        &state.show_dataset_panel);
        std::string systems_panel_shortcut = shortcut_label(ActionId::ToggleSystemsPanel);
        ImGui::MenuItem("Systems Panel", systems_panel_shortcut.empty() ? nullptr
                                                                        : systems_panel_shortcut.c_str(),
                        &state.show_systems_panel);
        std::string status_shortcut = shortcut_label(ActionId::ToggleStatusBar);
        ImGui::MenuItem("Status Strip", status_shortcut.empty() ? nullptr
                                                                : status_shortcut.c_str(),
                        &state.show_status_strip);
        std::string controls_shortcut = shortcut_label(ActionId::ToggleControls);
        ImGui::MenuItem("Sidebar Controls", controls_shortcut.empty() ? nullptr
                                                                      : controls_shortcut.c_str(),
                        &state.show_controls);
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Windows")) {
        if (show_sample_review) {
          std::string review_shortcut = shortcut_label(ActionId::ToggleSampleReview);
          ImGui::MenuItem("Sample Review", review_shortcut.empty() ? nullptr
                                                                   : review_shortcut.c_str(),
                          show_sample_review);
        }
        if (show_shortcuts_window) {
          std::string shortcuts_shortcut =
              shortcut_label(ActionId::ToggleShortcutsWindow);
          ImGui::MenuItem("Keyboard Shortcuts",
                          shortcuts_shortcut.empty() ? nullptr
                                                     : shortcuts_shortcut.c_str(),
                          show_shortcuts_window);
        }
        ImGui::EndMenu();
      }

      ImGui::Separator();
      std::string compact_shortcut = shortcut_label(ActionId::ToggleCompactUI);
      ImGui::MenuItem("Compact UI", compact_shortcut.empty() ? nullptr
                                                             : compact_shortcut.c_str(),
                      &state.compact_charts);
      std::string lock_shortcut = shortcut_label(ActionId::ToggleLockLayout);
      ImGui::MenuItem("Lock Layout", lock_shortcut.empty() ? nullptr
                                                           : lock_shortcut.c_str(),
                      &state.lock_layout);
      std::string reset_shortcut = shortcut_label(ActionId::ResetLayout);
      if (ImGui::MenuItem("Reset Layout",
                          reset_shortcut.empty() ? nullptr
                                                 : reset_shortcut.c_str())) {
        state.force_reset_layout = true;
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools")) {
        std::string auto_refresh_shortcut =
            shortcut_label(ActionId::ToggleAutoRefresh);
        ImGui::MenuItem("Auto Refresh",
                        auto_refresh_shortcut.empty()
                            ? nullptr
                            : auto_refresh_shortcut.c_str(),
                        &state.auto_refresh);
        std::string simulate_shortcut =
            shortcut_label(ActionId::ToggleSimulation);
        ImGui::MenuItem("Simulate Activity",
                        simulate_shortcut.empty()
                            ? nullptr
                            : simulate_shortcut.c_str(),
                        &state.simulate_activity);
        ImGui::Separator();
        std::string demo_shortcut = shortcut_label(ActionId::ToggleDemoWindow);
        ImGui::MenuItem("Show ImGui Demo", demo_shortcut.empty() ? nullptr
                                                                 : demo_shortcut.c_str(),
                        &state.show_demo_window);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("Documentation")) {}
      std::string help_shortcut =
          shortcut_label(ActionId::ToggleShortcutsWindow);
      if (ImGui::MenuItem("Shortcuts",
                          help_shortcut.empty() ? nullptr
                                                : help_shortcut.c_str())) {
        if (show_shortcuts_window) *show_shortcuts_window = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("About HAFS Viz")) {}
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void RenderSidebar(AppState& state, ImFont* font_ui, ImFont* font_header) {
  // Make the entire sidebar content scrollable to avoid overlaps
  ImGui::BeginChild("SidebarScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NoBackground);

  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 2));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.04f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.08f));

  auto sidebar_button = [&](const char* label, Workspace ws, const char* icon) {
    bool active = state.current_workspace == ws;
    if (active) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.85f, 1.0f, 1.0f));

    ImGui::PushID(label);
    ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, 40);
    if (ImGui::Button("##hidden", size)) {
      if (state.current_workspace != ws) {
        state.current_workspace = ws;
        if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
      }
    }
    
    ImVec2 p_min = ImGui::GetItemRectMin();
    ImVec2 p_max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    
    if (active) {
        draw->AddRectFilled(p_min, ImVec2(p_min.x + 3, p_max.y), ImColor(102, 217, 255));
        draw->AddRectFilled(p_min, p_max, ImColor(102, 217, 255, 10));
    }

    ImGui::SetCursorScreenPos(ImVec2(p_min.x + 15, p_min.y + 10));
    ImGui::BeginGroup();
    if (font_ui) ImGui::PushFont(font_ui);
    ImGui::Text("%s  %s", icon, label);
    if (font_ui) ImGui::PopFont();
    ImGui::EndGroup();

    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s workspace", label);
    ImGui::PopID();
    if (active) ImGui::PopStyleColor();
  };

  auto sidebar_header = [&](const char* title) {
    ImGui::Spacing(); ImGui::Spacing();
    if (font_header) ImGui::PushFont(font_header);
    ImGui::SetCursorPosX(15);
    ImGui::TextDisabled("%s", title);
    if (font_header) ImGui::PopFont();
    ImGui::Spacing();
  };

  sidebar_header("WORKSPACES");
  sidebar_button("Dashboard", Workspace::Dashboard, ICON_MD_DASHBOARD);
  sidebar_button("Analysis", Workspace::Analysis, ICON_MD_ANALYTICS);
  sidebar_button("Optimization", Workspace::Optimization, ICON_MD_SETTINGS_INPUT_COMPONENT);
  
  sidebar_header("OPERATIONS");
  sidebar_button("Systems", Workspace::Systems, ICON_MD_ROUTER);
  sidebar_button("Training", Workspace::Training, ICON_MD_MODEL_TRAINING);
  sidebar_button("Custom Grid", Workspace::Custom, ICON_MD_DASHBOARD_CUSTOMIZE);
  
  sidebar_header("REGISTRIES");
  sidebar_button("Chat", Workspace::Chat, ICON_MD_CHAT);
  sidebar_button("Context", Workspace::Context, ICON_MD_FOLDER_OPEN);
  sidebar_button("Models", Workspace::Models, ICON_MD_STICKY_NOTE_2);

  ImGui::PopStyleColor(3);
  ImGui::PopStyleVar();

  ImGui::EndChild(); // End SidebarScroll
}

} // namespace ui
} // namespace viz
} // namespace hafs
