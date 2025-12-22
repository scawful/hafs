#include "coverage_density.h"
#include "../core.h"
#include <vector>

namespace hafs::viz::ui {

void CoverageDensityChart::Render(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::CoverageDensity,
                    "DENSITY COVERAGE",
                    "Displays sample counts across latent space regions. Sparse regions (<50% of avg) indicate under-sampled scenarios.",
                    state);

  const auto& regions = loader.GetEmbeddingRegions();

  if (regions.empty()) {
    ImGui::TextDisabled("No embedding coverage data available");
    return;
  }

  // Scatter plot of region densities
  std::vector<float> dense_x, dense_y, sparse_x, sparse_y;
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

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##Coverage", state);
  if (ImPlot::BeginPlot("##Coverage", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Region Index", "Samples", axis_flags, axis_flags);
    if (state.show_plot_legends) {
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
    }
    HandlePlotContextMenu(PlotKind::CoverageDensity, state);
    
    // Low Density Zone Overlay
    double lx[2] = {-10, static_cast<double>(regions.size() + 10)};
    double ly1[2] = {0, 0};
    double ly2[2] = {avg * 0.5, avg * 0.5};
    ImPlot::SetNextFillStyle(ImVec4(1, 0.5f, 0, 0.1f));
    ImPlot::PlotShaded("Sparse Zone", lx, ly1, ly2, 2);

    ImVec4 healthy_color = GetSeriesColor(2);
    ImVec4 risk_color = GetSeriesColor(7);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, healthy_color);
    ImPlot::PlotScatter("Healthy", dense_x.data(), dense_y.data(), (int)dense_x.size());
    
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, risk_color);
    ImPlot::PlotScatter("At Risk", sparse_x.data(), sparse_y.data(), (int)sparse_x.size());
    
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

} // namespace hafs::viz::ui
