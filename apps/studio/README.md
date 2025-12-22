# hafs_studio

Native C++17 visualization and training management application for HAFS.

## Build

```bash
# From project root
cmake -B build -S . -DHAFS_BUILD_STUDIO=ON
cmake --build build --target hafs_studio
```

## Run

```bash
./build/apps/studio/hafs_studio
```

## Features

- **Dashboard**: Training metrics overview
- **Analysis**: Quality score trends, domain breakdown
- **Training Hub**: Real-time training status
- **Sample Review**: Data quality inspection
- **Text Editor**: Built-in code editor
- **Shortcut System**: Customizable keyboard shortcuts (Ctrl+/)

## Dependencies (auto-fetched)

- Dear ImGui (docking branch)
- ImPlot
- GLFW
- nlohmann/json
