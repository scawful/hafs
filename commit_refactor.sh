#!/bin/bash
cd /Users/scawful/Code/hafs
git add .
git commit -m "refactor: decouple TUI/CLI and promote framework utilities

- Move src/hafs/ui -> src/tui
- Move src/hafs/cli -> src/cli
- Move framework utilities (llm, adapters, plugins, etc.) to src/
- Establish backward-compatible shims in src/hafs/
- Optimize TUI performance (async loading, debouncing)
- Update global imports and pyproject.toml"
