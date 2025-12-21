# ML Training Data Visualization

HAFS includes visualization tools for monitoring ML training data quality, generator performance, embedding coverage, and training runs.

## TUI Dashboard (ASCII Charts)

The Training Dashboard is accessible from the main HAFS TUI via keybinding **9**.

### Features
- **Quality Trends**: Line chart showing quality metrics over recent samples
- **Generator Stats**: Bar chart of generator acceptance rates  
- **Embedding Coverage**: Scatter plot of region density (sparse vs dense)
- **Training Runs**: Bar chart comparing final loss across runs

### Usage
```bash
hafs  # Launch HAFS
# Press 9 to open Training Dashboard
# Press r to refresh data
# Press q to go back
```

### Data Sources
The dashboard reads from `~/.context/training/`:
- `quality_feedback.json` - Quality metrics and rejection history
- `active_learning.json` - Embedding region coverage data
- `training_feedback.json` - Training run metadata and losses

---

## Native GUI Application (ImGui/ImPlot)

For more interactive visualization, build the optional C++ application.

### Prerequisites
```bash
brew install glfw  # macOS
```

### Build
```bash
cd /Users/scawful/Code/hafs
cmake -B build/viz -S src/cc -DHAFS_BUILD_VIZ=ON
cmake --build build/viz
```

### Run
```bash
./build/viz/hafs_viz ~/.context/training
```

### Features
- Interactive charts with zoom/pan
- Dark theme matching HAFS aesthetic
- Menu bar with File > Refresh (F5)
- Status bar showing loaded data counts

---

## Data Format Reference

### quality_feedback.json
```json
{
  "generator_stats": {
    "AsmDataGenerator": {
      "samples_generated": 500,
      "samples_accepted": 425,
      "samples_rejected": 75,
      "rejection_reasons": {"low_diversity": 25, ...},
      "avg_quality_score": 0.82
    }
  },
  "rejection_history": [
    {"domain": "asm", "scores": {"diversity": 0.25, "overall": 0.58}}
  ]
}
```

### active_learning.json
```json
{
  "regions": [
    {"sample_count": 45, "domain": "asm", "avg_quality": 0.78}
  ],
  "num_regions": 50
}
```

### training_feedback.json
```json
{
  "training_runs": {
    "run_2025_12_20": {
      "model_name": "qwen2.5-coder-finetuned",
      "final_loss": 0.089,
      "samples_count": 1500
    }
  }
}
```
