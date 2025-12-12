# Halext Agentic File System (HAFS)

Context management tool for AI agents with a terminal user interface.

Based on research from: https://arxiv.org/pdf/2512.05470v1

## Features

- **Multi-Agent Orchestration** - Coordinate multiple AI agents with @mentions and content-based routing
- **Log Browsing** - Parse and search Gemini, Claude, and Antigravity session logs
- **AFS Project Management** - Organize files with policy-based access control (read_only, writable, executable)
- **Policy Overview** - View and edit AFS permissions directly from the dashboard
- **Workspace Browser** - Navigate filesystem and add files to context
- **Inline Viewer/Editor** - Preview markdown or raw text, edit and save inline, and rename/duplicate files without leaving the TUI
- **Planning/Execution Modes** - Toggle between strategic planning and implementation
- **Vim Navigation** - Optional vim-style keybindings (Ctrl+V to toggle)
- **Synergy Analysis** - Track collaboration quality between agents

## Installation

```bash
pip install hafs
```

Or install from source:

```bash
git clone https://github.com/scawful/hafs.git
cd hafs
pip install -e .
```

## Quick Start

```bash
# Launch the TUI dashboard
hafs

# Start multi-agent chat directly
hafs chat

# Browse AI session logs
hafs logs

# Show configuration
hafs config
```

## Keybindings (Which-Key / Spacemacs Style)

Press `SPC` (space) to open the which‑key bar, then follow the hints. Most actions are grouped under leader prefixes.

### Main Screen
- `SPC f` file actions: `a` add, `e` edit, `s` save, `r` rename, `d` delete, `c` copy, `o` open in OS
- `SPC p` projects/workspace: `w` add workspace, `u` edit global user context, `t` toggle sidebar, `[`/`]` resize sidebar
- `SPC s` search: `p` find file, `c` command palette
- `SPC c` context/chat: `c` chat with context selection, `x` add selection to context
- `SPC g` AI context generate
- `SPC l` logs
- `SPC q` quit

Direct keys still work for speed: `Ctrl+P` search, `Ctrl+K` palette, `Ctrl+B` toggle sidebar, `r` refresh, `q` quit.

### Chat (Orchestrator) Screen
- `SPC l` lanes: `1–4` focus lane, `n` next lane
- `SPC a` agents: `n` new agent, `k` kill agent
- `SPC t` toggles: `c` context panel, `s` synergy panel, `m` view mode
- `SPC p` permissions
- `SPC q` back

### Logs Screen
- `SPC t` tabs: `1` Gemini, `2` Antigravity, `3` Claude
- `SPC s` save selected session to context (pick mount)
- `SPC d` delete selected session
- `SPC r` refresh
- `SPC q` back

### Vim Mode (Ctrl+V to toggle)
| Key | Action |
|-----|--------|
| `j/k` | Move down/up |
| `h/l` | Collapse/expand |
| `gg` | Go to start |
| `G` | Go to end |
| `/` | Search |

## Configuration

HAFS loads configuration from:
1. `./hafs.toml` (project-local)
2. `~/.config/hafs/config.toml` (user)

Example configuration:

```toml
[general]
refresh_interval = 5
default_editor = "vim"

[[general.workspace_directories]]
path = "~/Code"
name = "Code"

[theme]
primary = "#4C3B52"
secondary = "#9B59B6"
accent = "#E74C3C"

[parsers.gemini]
enabled = true
base_path = "~/.gemini/tmp"

[[backends]]
name = "gemini"
enabled = true
command = ["gemini"]
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Multi-Agent System](docs/agents.md)
- [Agents Quick Start](docs/AGENTS_QUICKSTART.md)
- [AFS Policy Management](docs/AFS_POLICY_MANAGEMENT.md)
- [Improvement Ideas](docs/IMPROVEMENTS.md)

## Project Structure

```
hafs/
├── src/hafs/
│   ├── agents/      # Multi-agent orchestration
│   ├── backends/    # AI backend implementations
│   ├── config/      # Configuration management
│   ├── core/        # Parsers, search, AFS
│   ├── models/      # Pydantic data models
│   ├── synergy/     # Agent synergy analysis
│   └── ui/          # Textual TUI
├── docs/            # Documentation
└── tests/           # Test suite
```

## Requirements

- Python 3.11+
- Textual 0.40+
- Pydantic 2.0+
- rapidfuzz (for fuzzy search)

Optional:
- `gemini` CLI for Gemini backend
- `claude` CLI for Claude backend
- `tomli-w` for config saving

<img width="1800" height="1131" alt="Screenshot 2025-12-11 at 9 02 49 AM" src="https://github.com/user-attachments/assets/b4565d00-ce4a-4bbc-9ff7-8c9a01cda40a" />


## License

MIT

## Contributing

Contributions welcome! See [IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for ideas.
