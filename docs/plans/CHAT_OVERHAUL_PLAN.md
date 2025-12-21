# HAFS Chat Interface Overhaul Plan

## Objective
Transform the `hafs` TUI chat interface into a "High Performance" workspace that rivals modern TypeScript-based AI engineering tools (like `oracle-code` / `halext-code`), but implemented natively in Python/Textual for maximum integration and performance.

## Design Philosophy
1.  **Native & Fast**: No bridging to JS/TS. Pure Python optimized for Textual's render loop.
2.  **Session-Centric**: Work is organized into persistent "Sessions", not ephemeral chat streams.
3.  **Visual Density**: Information density should be high but organized (3-column layout).
4.  **Interactive Tooling**: Tool outputs are interactive widgets, not just text dumps.

## Architecture

### 1. The Workspace Layout (3-Column)
Move away from the "Split Lane" view to a unified workspace.

*   **Left Sidebar (Navigator)**:
    *   **Session Explorer**: Tree view of past/saved sessions.
    *   **Agent Roster**: Live status of active agents (CPU/Status indicators).
    *   **Context Tree**: Pinned files/docs for the current session.
*   **Center Stage (The Stream)**:
    *   **Unified Timeline**: User messages, Agent thoughts, Tool executions, and System events in a single chronological stream.
    *   **Virtual Scrolling**: Critical for performance. Re-implement `LaneContainer` to use a `ListView` or `DataTable` backed renderer for infinite scrolling without DOM weight.
    *   **Composer Area**: Multi-line, syntax-aware input at the bottom.
*   **Right Sidebar (Inspector)**:
    *   **Shared State**: Live view of `SharedContext` (facts, decisions).
    *   **Plan Tracker**: Dynamic checklist of the current mission.
    *   **Synergy Metrics**: Real-time "flow state" and IRT scores.

### 2. Performance Optimizations (Python/Textual Specific)
*   **Virtual DOM for Chat**: Instead of appending thousands of `Static` widgets (which kills Textual performance), use a custom `ListView` where items are rendered on-demand.
*   **Message Batching**: Agent streaming tokens should update an internal buffer, triggering a UI refresh only at 30-60fps, not on every token event.
*   **Background Workers**: Ensure all file I/O and Agent logic happens in `Worker` threads, communicating strictly via `call_from_thread` for UI updates.

### 3. Interactive Components
*   **Tool Cards**:
    *   **Pending**: "Approve/Deny" buttons.
    *   **Running**: Spinner + "Kill" button.
    *   **Done**: Summary view (collapsed) -> Click to expand full stdout/stderr.
*   **Command Palette (Cmd+K)**:
    *   Fuzzy finder for all actions (Switch Mode, Add Agent, Open File).
    *   Replaces many discrete keybindings.

### 4. Implementation Roadmap

#### Phase 1: Foundation & Layout
- [ ] Create `SessionWorkspace` widget (the new main container).
- [ ] Implement the 3-column grid layout using CSS grid for stability.
- [ ] Port `halext-code` theme tokens to `hafs.ui.theme`.

#### Phase 2: The High-Performance Stream
- [ ] Build `VirtualChatStream` widget (custom `ListView` or optimized container).
- [ ] Implement "Token Batcher" for smooth streaming updates.
- [ ] Migrate `StreamingMessage` to use this new architecture.

#### Phase 3: Interactive Tooling
- [ ] Create `ToolExecutionWidget`.
- [ ] Integrate with `AgentCoordinator`'s new `tool_confirmation_callback` to spawn UI requests directly in the stream.

#### Phase 4: Session Persistence
- [ ] Implement `SessionStore` (JSON/SQLite backed).
- [ ] Add Save/Load commands to the sidebar.

## Migration Strategy
We will build `hafs/ui/screens/workspace.py` alongside the existing `chat.py`. Once feature parity and performance targets are met, we will swap the route.
