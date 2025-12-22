# C++ Visualization App Refactoring Proposal

**Date:** 2025-12-22
**Current Name:** hafs_viz
**Current Location:** `src/cc/viz/`

---

## Executive Summary

The C++ visualization application has grown organically and now deserves:
1. **Better naming** - "hafs_viz" is too generic
2. **Proper structure** - Move from `src/cc` to dedicated location
3. **Feature expansion** - Integrate model registry, training monitoring, cross-platform ops
4. **Modern architecture** - Better separation of concerns

---

## Current State Analysis

### Structure

```
src/cc/
├── viz/                           # Main app (1,392 lines)
│   ├── app.cc (624 lines)        # Main application logic
│   ├── data_loader.cc (723 lines) # Training data loading
│   ├── main.cc (45 lines)        # Entry point
│   ├── widgets/                   # UI components
│   │   ├── training_status.cpp
│   │   ├── sample_review.cc
│   │   ├── text_editor.cc (133k - large!)
│   │   └── imgui_memory_editor.h
│   ├── ui/                        # UI infrastructure
│   │   ├── core.cc
│   │   ├── shortcuts.cc
│   │   └── components/
│   ├── models/                    # Data models
│   │   └── state.h
│   ├── themes/                    # Visual themes
│   │   └── hafs_theme.h
│   └── assets/                    # Fonts, icons
│       └── font/
├── embed/                         # Embedding operations
├── similarity/                    # Similarity search
├── quantize/                      # Model quantization
├── index/                         # Vector indexing
├── io/                           # I/O operations
└── bindings/                      # Python bindings
```

### Features

**Current:**
- Training data visualization
- Sample quality review
- Real-time metrics (simulated)
- Multiple workspace views (Dashboard, Analysis, Optimization, etc.)
- Text editor integration
- Memory editor for binary data
- Shortcut management
- Theme system

**Missing:**
- Model registry integration
- Cross-platform training status (Windows GPU)
- Model deployment controls
- Real-time training logs
- Model comparison tools
- Dataset quality analytics
- Agent orchestration visualization

---

## Problems

### 1. Naming

**Current:** `hafs_viz`

**Issues:**
- Too generic
- Doesn't convey purpose
- Not memorable

**Alternatives:**
- `hafs_insight` - Emphasizes analytics/intelligence
- `hafs_studio` - IDE-like feel
- `hafs_command` - Command center feel
- `hafs_mission_control` - Training mission control
- `hafs_dashboard` - Simple, clear
- `hafs_oracle_studio` - Oracle model development focus

**Recommendation:** `hafs_studio` or `hafs_insight`

### 2. Location

**Current:** `src/cc/`

**Issues:**
- Mixed with C++ native modules (embed, similarity, etc.)
- "cc" doesn't convey "C++" (historical name)
- No clear separation between library and application

**Proposed Structure:**

```
apps/
└── studio/                        # Main GUI app (rename from viz)
    ├── src/
    │   ├── main.cc
    │   ├── app.{h,cc}
    │   ├── core/                  # Core app logic
    │   │   ├── data_loader.{h,cc}
    │   │   ├── state_manager.{h,cc}
    │   │   └── config.{h,cc}
    │   ├── ui/                    # UI framework
    │   │   ├── core.{h,cc}
    │   │   ├── shortcuts.{h,cc}
    │   │   ├── theme.h
    │   │   └── components/
    │   │       ├── training_dashboard.{h,cc}
    │   │       ├── model_registry.{h,cc}
    │   │       ├── deployment_panel.{h,cc}
    │   │       └── log_viewer.{h,cc}
    │   ├── widgets/               # Reusable widgets
    │   │   ├── text_editor/
    │   │   ├── memory_editor/
    │   │   ├── sample_review/
    │   │   ├── chart_view/
    │   │   └── terminal/
    │   ├── models/                # Data models
    │   │   ├── training.h
    │   │   ├── registry.h
    │   │   └── deployment.h
    │   └── utils/                 # Utilities
    │       ├── json.h
    │       ├── path.h
    │       └── logger.h
    ├── assets/
    │   ├── fonts/
    │   ├── icons/
    │   └── themes/
    ├── CMakeLists.txt
    └── README.md

native/                            # C++ library modules (rename from cc)
├── embed/                         # Embedding operations
├── similarity/                    # Similarity search
├── quantize/                      # Model quantization
├── index/                         # Vector indexing
├── io/                           # I/O operations
└── bindings/                      # Python bindings
    └── CMakeLists.txt
```

### 3. Code Organization

**Issues:**
- `app.cc` is 624 lines - getting large
- `data_loader.cc` is 723 lines - should be split
- `text_editor.cc` is 133k lines - vendored dependency, should be in `third_party/`
- Mixed concerns (UI + business logic)

**Improvements:**
- Split `data_loader.cc` into domain-specific loaders
- Extract view rendering into separate files
- Better separation: Models ↔ Controllers ↔ Views
- Move large vendored deps to `third_party/`

---

## Proposed New Features

### 1. Model Registry Integration

**Component:** `ui/components/model_registry.{h,cc}`

**Features:**
- List registered models from `~/.context/models/registry.json`
- Show model metadata (loss, dataset, hardware)
- Quick model selection for deployment
- Model comparison view

**UI:**
```
╭─ Model Registry ─────────────────────────────────────────╮
│ ┌─────────────────────────────────────────────────────┐  │
│ │ oracle-rauru-asm-qwen25-coder-15b-20251222         │  │
│ │   Role: asm  │  Loss: 0.5855  │  Windows GPU       │  │
│ │   [Pull] [Deploy] [Test] [Compare]                 │  │
│ ├─────────────────────────────────────────────────────┤  │
│ │ oracle-farore-general-qwen25-coder-15b-20251222    │  │
│ │   Role: general  │  Training: 35%  │  ETA: 48min   │  │
│ │   [View Logs] [Stop]                               │  │
│ └─────────────────────────────────────────────────────┘  │
╰──────────────────────────────────────────────────────────╯
```

### 2. Cross-Platform Training Monitor

**Component:** `ui/components/training_dashboard.{h,cc}`

**Features:**
- Real-time status from Windows GPU via mount/SSH
- Training progress visualization
- Loss curves
- GPU utilization
- Dataset quality metrics
- Auto-refresh with configurable interval

**Integration:**
- Use `CrossPlatformPath` from Python (expose via JSON API)
- Poll `trainer_state.json` from Windows mount
- Parse training logs for real-time metrics

### 3. Deployment Panel

**Component:** `ui/components/deployment_panel.{h,cc}`

**Features:**
- One-click model deployment
- Format conversion (PyTorch → GGUF)
- Quantization selection
- Backend selection (Ollama, llama.cpp, halext)
- Test prompt execution
- Status display

**UI:**
```
╭─ Model Deployment ────────────────────────────────────╮
│ Model: oracle-rauru-asm-qwen25-coder-15b-20251222    │
│                                                       │
│ Backend: [Ollama ▼]                                  │
│ Format:  [GGUF ▼]                                    │
│ Quant:   [Q4_K_M ▼]                                  │
│                                                       │
│ ┌─ Test Prompt ─────────────────────────────────┐    │
│ │ Write a JSR instruction                       │    │
│ └───────────────────────────────────────────────┘    │
│                                                       │
│ [Convert] [Deploy] [Test]                            │
│                                                       │
│ Status: Ready to deploy                              │
╰───────────────────────────────────────────────────────╯
```

### 4. Live Log Viewer

**Component:** `ui/components/log_viewer.{h,cc}`

**Features:**
- Tail training logs in real-time
- Color-coded by log level
- Search/filter
- Follow mode (auto-scroll)
- Export selection

**Sources:**
- `/tmp/oracle_farore_training.log`
- `~/.context/training/logs/`
- Windows mount: `D:/hafs_training/logs/`

### 5. Dataset Quality Analytics

**Component:** `ui/components/dataset_analytics.{h,cc}`

**Features:**
- Acceptance/rejection rates
- Diversity score distribution
- Domain breakdown
- Quality score trends
- Rejection reasons (pie chart)
- Sample browser with filtering

### 6. Agent Orchestration View

**Component:** `ui/components/agent_orchestrator.{h,cc}`

**Features:**
- Running agents status
- Task queue
- Agent performance metrics
- Error logs
- Manual task submission

---

## Implementation Plan

### Phase 1: Restructure (Week 1)

1. **Rename app** ✓ Decide on name (hafs_studio)
2. **Move directories:**
   ```bash
   mkdir -p apps/studio/src
   mkdir -p native

   # Move viz → studio
   mv src/cc/viz/* apps/studio/src/

   # Move native modules
   mv src/cc/{embed,similarity,quantize,index,io,bindings} native/

   # Move large vendor deps
   mkdir -p third_party/imgui
   mv apps/studio/src/widgets/text_editor.{cc,h} third_party/text_editor/
   ```

3. **Update CMake:**
   - `apps/studio/CMakeLists.txt` - Main app build
   - `native/CMakeLists.txt` - Native library modules
   - Root `CMakeLists.txt` - Top-level orchestration

4. **Update imports/includes**

5. **Test build:**
   ```bash
   cmake -B build -S . -DHAFS_BUILD_STUDIO=ON
   cmake --build build
   ```

### Phase 2: Model Registry Integration (Week 2)

1. **Create JSON reader:**
   ```cpp
   // apps/studio/src/utils/json_reader.h
   class RegistryReader {
     std::vector<ModelMetadata> LoadRegistry(const std::filesystem::path& path);
   };
   ```

2. **Create model registry UI:**
   ```cpp
   // apps/studio/src/ui/components/model_registry.{h,cc}
   class ModelRegistryWidget {
     void Render();
     void OnPullModel(const std::string& model_id);
     void OnDeployModel(const std::string& model_id);
   };
   ```

3. **Integrate into main app:**
   ```cpp
   // apps/studio/src/app.cc
   void App::RenderModelRegistryView() {
     model_registry_widget_.Render();
   }
   ```

### Phase 3: Training Monitor (Week 3)

1. **Create cross-platform path helper:**
   ```cpp
   // apps/studio/src/utils/path.h
   std::filesystem::path ResolveMountPath(const std::string& remote_path);
   bool CheckMount(const std::filesystem::path& mount_point);
   ```

2. **Create training state parser:**
   ```cpp
   // apps/studio/src/core/training_monitor.{h,cc}
   class TrainingMonitor {
     TrainingStatus PollWindows();
     TrainingStatus PollLocal();
     std::vector<MetricPoint> LoadLossCurve();
   };
   ```

3. **Create real-time dashboard:**
   ```cpp
   // apps/studio/src/ui/components/training_dashboard.{h,cc}
   class TrainingDashboard {
     void RenderProgressBar();
     void RenderLossCurve();
     void RenderGPUStats();
     void RenderDatasetMetrics();
   };
   ```

### Phase 4: Deployment Tools (Week 4)

1. **Create deployment controller:**
   ```cpp
   // apps/studio/src/core/deployment.{h,cc}
   class DeploymentController {
     void ConvertToGGUF(const std::string& model_id, const std::string& quant);
     void DeployToOllama(const std::string& model_id);
     std::string TestModel(const std::string& prompt);
   };
   ```

2. **Create deployment UI:**
   ```cpp
   // apps/studio/src/ui/components/deployment_panel.{h,cc}
   class DeploymentPanel {
     void RenderBackendSelector();
     void RenderFormatSelector();
     void RenderTestPrompt();
     void OnDeploy();
   };
   ```

### Phase 5: Polish & Release (Week 5)

1. **Add log viewer**
2. **Add dataset analytics**
3. **Update documentation**
4. **Create release binary**
5. **Integration tests**

---

## Technical Decisions

### 1. JSON Parsing

**Options:**
- **nlohmann/json** (header-only, modern C++)
- **RapidJSON** (fast, SAX/DOM)
- **simdjson** (fastest, SIMD)

**Recommendation:** **nlohmann/json** for ease of use

### 2. HTTP Client (for API calls)

**Options:**
- **cpr** (libcurl wrapper, modern C++)
- **httplib** (header-only)
- **CURL** (C library)

**Recommendation:** **httplib** for simplicity

### 3. Process Execution

**For running hafs CLI commands:**

**Options:**
- `std::system()` (simple but limited)
- `popen()` (capture output)
- Boost.Process (full-featured)

**Recommendation:** `popen()` with proper error handling

### 4. Configuration

**Current:** None (hardcoded paths)

**Proposed:** TOML config file

```toml
# apps/studio/config.toml
[paths]
registry = "~/.context/models/registry.json"
training = "~/.context/training"
windows_mount = "~/Mounts/mm-d"

[windows]
enabled = true
host = "medical-mechanica"
training_path = "D:/hafs_training"

[ui]
theme = "dark"
font_size = 14
refresh_interval = 5000  # ms
```

**Library:** **toml++** (header-only, modern C++)

---

## Benefits

### For Development
- Clear separation: App vs Native Library
- Easier to navigate
- Better code organization
- Vendor deps isolated

### For Users
- Professional name (`hafs_studio`)
- All-in-one training/deployment tool
- Real-time cross-platform monitoring
- One-click model deployment
- Better UX with integrated workflows

### For Maintenance
- Modular architecture
- Easier testing
- Clear responsibilities
- Documentation aligned with structure

---

## Migration Path

### Backward Compatibility

**Build:**
```cmake
# Old way (still works)
cmake -B build -S src/cc -DHAFS_BUILD_VIZ=ON

# New way
cmake -B build -S . -DHAFS_BUILD_STUDIO=ON
```

**Binary:**
```bash
# Old name (symlink)
./build/hafs_viz

# New name
./build/hafs_studio
```

### Documentation Updates

- Update `docs/training/` references
- Add `docs/studio/` with user guide
- Update README.md build instructions
- Add migration guide

---

## Estimated Effort

| Phase | Duration | Complexity |
|-------|----------|------------|
| Restructure | 1 week | Medium |
| Model Registry | 1 week | Low |
| Training Monitor | 1 week | Medium |
| Deployment Tools | 1 week | High |
| Polish | 1 week | Low |
| **Total** | **5 weeks** | **Medium** |

---

## Risks & Mitigation

### Risk 1: Build System Changes

**Impact:** Build could break
**Mitigation:** Keep old CMake targets as aliases during transition

### Risk 2: Large Codebase Move

**Impact:** Git history fragmented
**Mitigation:** Use `git mv` to preserve history

### Risk 3: Cross-Platform Testing

**Impact:** Windows paths might not work
**Mitigation:** Mock mount point for testing, comprehensive path tests

---

## Recommendations

### Immediate Actions

1. **Rename to `hafs_studio`** - Clear, professional
2. **Create `apps/` and `native/` structure** - Better organization
3. **Add model registry integration** - High value, low effort
4. **Add real-time training monitor** - Critical for Windows GPU workflow

### Future Enhancements

- Web UI (Electron/Tauri wrapper)
- Remote deployment (deploy to halext nodes)
- Model fine-tuning UI
- Dataset augmentation tools
- Agent conversation replay
- Performance profiling

---

## Conclusion

The C++ visualization app has outgrown its initial scope. Rebranding as **hafs_studio** and restructuring the codebase will:

1. Improve maintainability
2. Enable rapid feature development
3. Provide better user experience
4. Align with professional standards

**Next Step:** Approve naming and structure, then begin Phase 1 (Restructure).
