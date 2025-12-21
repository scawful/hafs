# HAFS Agentic System Guide

This document provides a comprehensive overview of the HAFS agentic workflow system, including the modular CLI, interactive shell, safety tooling, and the high-performance TUI Session Workspace.

## 1. Modular CLI Architecture (`src/hafs/cli/`)

The HAFS CLI is structured as a modular package to ensure maintainability and extensibility.

- **`main.py`**: The central entry point. It registers all subcommand groups and handles plugin discovery.
- **`commands/`**: Contains discrete modules for each CLI category:
    - `afs.py`: Agentic File System management (`init`, `mount`, `list`).
    - `chat.py`: Interactive agent shell (REPL).
    - `orchestrator.py`: Plan-Execute-Verify pipelines.
    - `services.py`: Background service management.
    - `history.py`, `embed.py`, `memory.py`, `nodes.py`, `sync.py`: Specialized infrastructure commands.

### Key UX Features
- **Help-on-Empty**: Running a command or category without arguments (e.g., `hafs orchestrate`) automatically displays its help menu.
- **Zsh Autocompletion**: Installable via `hafs --install-completion zsh` or `./scripts/install_zsh_completion.sh`.

---

## 2. Interactive Agent Shell (`hafs chat`)

The `chat` command (aliased as `shell`) provides a persistent, multi-turn interactive session with the `AgentCoordinator`.

### Workflow
1.  **Routing**: User input is routed to the most appropriate agent based on the request.
2.  **Streaming**: Agent responses are streamed in real-time to the console.
3.  **Context Persistence**: The session maintains a shared context across turns, allowing for complex, multi-step reasoning.

---

## 3. Safety Tooling & Sandbox (`hafs.core.tooling`)

HAFS implements a "Safety-First" approach to agentic tool execution.

### Components
- **`ToolProfile`**: Defines `allow`, `deny`, and `requires_confirmation` lists for tools.
- **`ToolRunner`**: Executes commands within a project root, enforcing the profile. It prevents shell expansion and ensures tools run in a controlled environment.
- **Interactive Confirmation**: Sensitive tool categories (e.g., `write`, `deploy`) trigger a confirmation prompt in the interactive shell. The agent cannot proceed until the user explicitly approves the action.

### Agent Execution Protocol
Agents request tool use via a structured XML block:
```xml
<execute>
git status --porcelain
</execute>
```
The system parses this, checks the safety profile, requests confirmation if needed, and feeds the output back into the agent's context.

---

## 4. TUI Session Workspace

The TUI (`hafs`) features a high-performance **Session Workspace** (Key `5`) designed for deep engineering tasks.

### 3-Column Layout
1.  **Navigator (Left)**:
    - **Session Explorer**: Manage persistent session documents.
    - **Agent Roster**: Monitor live agent status and health.
    - **Context Tree**: Browse the active AFS structure.
2.  **Stage (Center)**:
    - **VirtualChatStream**: A high-performance renderer that archives old messages to keep the UI responsive.
    - **Composer**: Multi-line input for agent interaction.
3.  **Inspector (Right)**:
    - **Shared State**: Real-time view of facts, decisions, and findings.
    - **Plan Tracker**: Live checklist of the current mission's progress.

### Session Persistence (`src/hafs/core/sessions.py`)
Sessions are saved as JSON documents in `~/.context/sessions/`, capturing:
- Metadata (name, agents, timestamps).
- Full message history.
- The state of the `SharedContext`.

---

## 5. Summary for Future Agents

When working on this system:
- **CLI**: Add new commands to `src/hafs/cli/commands/` and register them in `main.py`.
- **UI**: Use `src/hafs/ui/widgets/` for reusable components. Ensure performance by using `VirtualChatStream` for long-running logs.
- **Safety**: Always execute external commands via `ToolRunner` to ensure they respect the user's safety profile.
- **Context**: Leverage the `SharedContext` in `src/hafs/models/agent.py` to share state between agents.
