#pragma once

#include "chart.h"

namespace hafs::viz::ui {

class QualityTrendsChart : public Chart {
public:
    void Render(AppState& state, const DataLoader& loader) override;
    std::string GetTitle() const override { return "Quality Trends"; }
    PlotKind GetKind() const override { return PlotKind::QualityTrends; }
};

} // namespace hafs::viz::ui
