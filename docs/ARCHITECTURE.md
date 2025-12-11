# HAFS Architecture Documentation

## Overview

HAFS (Halext Agentic File System) is a context management tool for AI agents, featuring a TUI (Terminal User Interface) for browsing logs, managing multi-agent orchestration, and organizing project contexts.

Based on research from: https://arxiv.org/pdf/2512.05470v1

## Project Structure

```
hafs/
├── src/hafs/
│   ├── __main__.py          # Entry point
│   ├── cli.py               # CLI commands (hafs, hafs chat, hafs logs, etc.)
│   │
│   ├── agents/              # Multi-agent orchestration
│   │   ├── __init__.py      # Package exports
│   │   ├── coordinator.py   # AgentCoordinator - high-level orchestration
│   │   ├── lane.py          # AgentLane - individual agent execution
│   │   ├── roles.py         # Role definitions and keyword matching
│   │   └── router.py        # MentionRouter - @mention and content routing
│   │
│   ├── backends/            # AI backend implementations
│   │   ├── __init__.py      # BackendRegistry
│   │   ├── base.py          # BaseBackend abstract class
│   │   ├── claude.py        # ClaudeCliBackend
│   │   └── gemini.py        # GeminiCliBackend
│   │
│   ├── config/              # Configuration management
│   │   ├── __init__.py
│   │   ├── loader.py        # Load config from TOML files
│   │   ├── saver.py         # Save config to TOML files
│   │   └── schema.py        # Pydantic config models
│   │
│   ├── context/             # Context building utilities
│   │   ├── __init__.py
│   │   └── builder.py       # ContextBuilder for prompt assembly
│   │
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── search.py        # Fuzzy search utilities (rapidfuzz)
│   │   ├── afs/             # Agentic File System
│   │   │   ├── discovery.py # Project discovery
│   │   │   ├── manager.py   # AFS manager
│   │   │   └── policy.py    # Policy enforcement
│   │   ├── parsers/         # Log parsers
│   │   │   ├── base.py      # BaseParser abstract class
│   │   │   ├── registry.py  # ParserRegistry
│   │   │   ├── gemini.py    # GeminiLogParser
│   │   │   ├── claude.py    # ClaudePlanParser
│   │   │   └── antigravity.py # AntigravityParser
│   │   └── services/        # Background services
│   │
│   ├── models/              # Data models (Pydantic)
│   │   ├── __init__.py
│   │   ├── agent.py         # Agent, AgentMessage, AgentRole, SharedContext
│   │   ├── afs.py           # ContextRoot, Mount, MountType
│   │   ├── gemini.py        # GeminiSession, GeminiMessage
│   │   ├── claude.py        # PlanDocument, TaskStatus
│   │   ├── antigravity.py   # AntigravityBrain, AntigravityTask
│   │   └── synergy.py       # SynergyScore, AgentProfile
│   │
│   ├── plugins/             # Plugin system
│   │   ├── __init__.py
│   │   ├── loader.py        # Plugin discovery and loading
│   │   └── protocol.py      # Plugin protocol definition
│   │
│   ├── synergy/             # Agent synergy analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py      # SynergyAnalyzer
│   │   ├── evaluator.py     # Response evaluation
│   │   ├── markers.py       # Collaboration markers
│   │   ├── profile.py       # Agent profiles
│   │   └── scoring.py       # Synergy scoring
│   │
│   ├── adapters/            # External adapters
│   │   ├── __init__.py
│   │   └── base.py          # Base adapter class
│   │
│   └── ui/                  # Textual TUI
│       ├── __init__.py
│       ├── app.py           # HafsApp - main application
│       ├── theme.py         # HalextTheme - purple gradient theme
│       │
│       ├── mixins/          # Reusable mixins
│       │   └── vim_navigation.py  # VimNavigationMixin
│       │
│       ├── screens/         # TUI screens
│       │   ├── main.py              # MainScreen - dashboard
│       │   ├── orchestrator.py      # OrchestratorScreen - multi-agent chat
│       │   ├── logs.py              # LogsScreen - log browser
│       │   ├── settings.py          # SettingsScreen
│       │   ├── command_palette.py   # CommandPalette modal
│       │   ├── context_selection_modal.py  # Context picker
│       │   ├── file_picker_modal.py # File browser modal
│       │   ├── permissions_modal.py # AFS policy editor
│       │   ├── ai_context_modal.py  # AI context generation
│       │   ├── input_modal.py       # Generic input modal
│       │   └── help_modal.py        # Help screen
│       │
│       └── widgets/         # Reusable widgets
│           ├── agent_lane.py        # AgentLaneWidget
│           ├── chat_input.py        # ChatInput with autocomplete
│           ├── context_panel.py     # Context display panel
│           ├── context_viewer.py    # File/project viewer
│           ├── filesystem_tree.py   # Workspace browser
│           ├── header_bar.py        # App header
│           ├── keybinding_bar.py    # Two-row keybinding display
│           ├── lane_container.py    # Agent lanes container
│           ├── mode_toggle.py       # Planning/Execution toggle
│           ├── plan_viewer.py       # Claude plans viewer
│           ├── project_tree.py      # AFS project tree
│           ├── session_detail.py    # Log session detail
│           ├── session_list.py      # Log session list
│           ├── sidebar_panel.py     # Collapsible sidebar
│           ├── split_log_view.py    # Split list/detail view
│           ├── stats_panel.py       # Statistics display
│           ├── synergy_panel.py     # Synergy score display
│           └── afs_control.py       # AFS controls
│
├── docs/                    # Documentation
├── tests/                   # Test suite
└── pyproject.toml           # Project configuration
```

## Core Components

### 1. CLI (`cli.py`)

Entry points for the application:

- `hafs` / `hafs tui` - Launch the TUI dashboard
- `hafs chat` - Launch directly into multi-agent chat
- `hafs logs` - Browse AI session logs
- `hafs config` - Show configuration
- `hafs version` - Show version info

### 2. Configuration System

**Schema** (`config/schema.py`):
- `HafsConfig` - Root configuration
- `GeneralConfig` - General settings (refresh interval, editor, workspace directories)
- `ThemeConfig` - UI theme colors
- `ParserConfig` - Parser settings (Gemini, Claude, Antigravity)
- `AFSDirectoryConfig` - AFS policy definitions
- `BackendConfig` - AI backend configuration

**Loader** (`config/loader.py`):
- Loads from `./hafs.toml` (project-local) or `~/.config/hafs/config.toml` (user)
- Merges configurations with sensible defaults

**Saver** (`config/saver.py`):
- Persists configuration changes to TOML
- Handles Path serialization and enum conversion

### 3. Multi-Agent System

**Coordinator** (`agents/coordinator.py`):
- Manages multiple AI agents
- Routes messages via @mentions or content analysis
- Maintains shared context across agents
- Supports planning and execution modes

**Agent Lane** (`agents/lane.py`):
- Individual agent execution context
- Message queue management
- Streaming response handling
- Context injection

**Router** (`agents/router.py`):
- Extracts @mentions from messages
- Routes by keywords to appropriate agent roles
- Falls back to general agent

**Roles** (`agents/roles.py`):
- GENERAL, PLANNER, CODER, CRITIC, RESEARCHER
- Role-specific system prompts
- Keyword-to-role mapping

### 4. Backend System

**Registry** (`backends/__init__.py`):
- `BackendRegistry.register(backend_class)` - Register backend
- `BackendRegistry.create(name)` - Instantiate backend
- `BackendRegistry.list_available()` - List registered backends

**Backends**:
- `GeminiCliBackend` - Interfaces with Gemini CLI
- `ClaudeCliBackend` - Interfaces with Claude CLI
- Uses PTY for streaming output

### 5. Parser System

**Registry** (`core/parsers/registry.py`):
- `ParserRegistry.register(name, parser_class)`
- `ParserRegistry.get(name)` - Get parser class

**Parsers**:
- `GeminiLogParser` - Parses `~/.gemini/tmp/` sessions
- `ClaudePlanParser` - Parses `~/.claude/plans/` markdown
- `AntigravityParser` - Parses Antigravity brain files

### 6. AFS (Agentic File System)

**Discovery** (`core/afs/discovery.py`):
- Scans for `.context/` directories
- Builds `ContextRoot` models with mounts

**Policy** (`core/afs/policy.py`):
- Enforces read_only, writable, executable policies
- Validates file operations against policies

**Mount Types**:
- `MEMORY` - Long-term storage (read_only)
- `KNOWLEDGE` - Reference materials (read_only)
- `TOOLS` - Executable scripts (executable)
- `SCRATCHPAD` - Working space (writable)
- `HISTORY` - Historical data (read_only)

### 7. TUI System

**App** (`ui/app.py`):
- `HafsApp` extends Textual's `App`
- Manages screen stack and navigation
- Initializes coordinator for chat mode

**Theme** (`ui/theme.py`):
- `HalextTheme` - Purple gradient theme
- Generates TCSS variables for consistent styling

**Screens**:
- `MainScreen` - Dashboard with project tree, workspace browser, stats
- `OrchestratorScreen` - Multi-agent chat interface
- `LogsScreen` - Tabbed log browser (Gemini, Antigravity, Claude)
- `SettingsScreen` - Configuration display and editing

**Key Widgets**:
- `ChatInput` - Input with slash command autocomplete
- `AgentLaneWidget` - Individual agent output display
- `FilesystemTree` - Workspace directory browser
- `ProjectTree` - AFS project structure
- `KeyBindingBar` - Two-row keybinding display

### 8. Synergy System

Analyzes collaboration quality between agents:

- `SynergyAnalyzer` - Tracks agent interactions
- `SynergyScore` - Quantifies collaboration effectiveness
- Markers for handoffs, agreements, building on ideas

## Data Flow

```
User Input
    │
    ▼
┌─────────────┐
│  ChatInput  │ ──── Slash commands (/add, /task, etc.)
└─────────────┘
    │
    ▼
┌─────────────────┐
│ AgentCoordinator │
└─────────────────┘
    │
    ├── @mention ──► MentionRouter ──► Target Agent
    │
    └── No mention ──► Content Analysis ──► Best Role Agent
                              │
                              ▼
                      ┌─────────────┐
                      │  AgentLane  │
                      └─────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │   Backend   │ (Gemini/Claude CLI)
                      └─────────────┘
                              │
                              ▼
                      Streaming Response
                              │
                              ▼
                      ┌───────────────────┐
                      │ AgentLaneWidget   │
                      └───────────────────┘
```

## Configuration Files

### `hafs.toml` (Project-local)

```toml
[general]
refresh_interval = 5
default_editor = "vim"
show_hidden_files = false

[[general.workspace_directories]]
path = "~/Code"
name = "Code"
recursive = true

[theme]
primary = "#4C3B52"
secondary = "#9B59B6"
accent = "#E74C3C"

[parsers.gemini]
enabled = true
base_path = "~/.gemini/tmp"

[[afs_directories]]
name = "memory"
policy = "read_only"
description = "Long-term memory storage"

[[backends]]
name = "gemini"
enabled = true
command = ["gemini"]
```

### `.context/metadata.json` (Project AFS)

```json
{
  "created_at": "2025-01-01T00:00:00",
  "description": "Project description",
  "agents": ["planner", "coder"],
  "policy": {
    "read_only": ["memory", "knowledge"],
    "writable": ["scratchpad"],
    "executable": ["tools"]
  }
}
```

## Key Design Patterns

### 1. Registry Pattern
Used for backends and parsers to allow dynamic registration and lookup.

### 2. Mixin Pattern
`VimNavigationMixin` adds vim keybindings to any screen.

### 3. Message Passing
Textual's message system for widget communication.

### 4. Async Streaming
PTY-based streaming for real-time AI responses.

### 5. Lazy Loading
File trees load directory contents on expansion.

### 6. Modal Screens
Overlay modals for focused interactions.
