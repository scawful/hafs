# Chat Mode + Rendering Engine Feasibility Plan

This plan explores a dedicated chat mode and cross-language rendering options
to improve performance and UX for multi-agent swarms/councils. It is a research
and prototyping roadmap, not a commitment to replace the current UI.

## Goals

- Low-latency, high-throughput chat UX for multi-agent orchestration.
- Streaming updates with clear agent attribution and tool result panels.
- A renderer architecture that can scale across macOS, Windows, and server nodes.
- A clean adapter boundary so the current Textual UI and any future UI can
  integrate without blocking each other.

## Non-Goals

- Replacing the current UI immediately.
- Rewriting core orchestration logic in another language.
- Duplicating the UI overhaul that is already in progress.

## UX Requirements (Chat Mode)

- Streaming token display with minimal jank (no full-screen reflow per token).
- Multi-agent lanes or grouped threads (per council lane or per swarm).
- Inline tool output cards (logs, diffs, metrics, graphs).
- Quick actions (rerun, inspect, copy, pin, open docs).
- Fast search and jump-to-message.
- Timeline markers for phases (plan → execute → verify → summarize).

## Performance Targets (Initial)

- Input-to-render latency < 50ms for chat events.
- Target 30+ FPS for TUI; 60 FPS for GPU-backed UI.
- Token streaming throughput > 50 tokens/sec without layout stalls.

## Architecture Direction

### 1. UI Adapter Protocol (Core Boundary)

Define a renderer-agnostic adapter protocol that receives events from HAFS:

- `ChatEvent` (message content, role, agent id, timestamps, tags)
- `AgentStatus` (thinking, executing, idle, health, node)
- `ToolResult` (stdout, stderr, artifacts, durations)
- `PhaseEvent` (plan/execute/verify/summarize milestones)
- `MetricsUpdate` (tokens, latency, costs, eval scores)

Transport options:

- Local IPC (Unix sockets) for native renderers
- gRPC/protobuf for cross-language bindings
- WebSocket for web/native hybrid UI

### 2. Rendering Engine Candidates

- **Textual (Python)**: keep as baseline; optimize with virtualization and
  incremental layout updates.
- **Rust (ratatui/crossterm)**: strong terminal performance; integrate via
  IPC/JSON-RPC; good for CLI-first environments.
- **C++ (ImGui/Skia/SDL)**: GPU-backed rendering; fast and flexible for rich UX.
- **SwiftUI (macOS)** + **WinUI (Windows)**: native look, higher maintenance.
- **Web UI (Tauri/React)**: best for rich visuals; reuse existing web dashboard.

### 3. Proposed Integration Pattern

- HAFS core remains Python.
- Renderer runs as a separate process and subscribes to UI events.
- A small “renderer bridge” publishes events and manages backpressure.
- UI adapter interface is stable; multiple renderers can coexist.

## Feasibility Work (Phase CM0)

- Profile current Textual UI with high-volume chat logs.
- Identify bottlenecks: layout thrash, widget count, token streaming.
- Baseline memory usage and CPU impact during multi-agent streams.
- Produce a minimal event protocol draft and message schema.

## Prototype Roadmap

### Phase CM1: Chat Mode Event Core

- Implement event stream in core (UI-agnostic).
- Add a `chat_mode` entrypoint (`hafs chat`) that emits events.
- Build a simple Textual adapter using the new stream.

### Phase CM2: Cross-Language Renderer Prototype

- Choose one native renderer to prototype (Rust or C++).
- Implement a small receiver that renders:
  - streaming messages,
  - agent lanes,
  - tool output cards.
- Validate throughput and latency against baseline.

### Phase CM3: UX Enhancements

- Add search, quick actions, and phase markers.
- Add lane filtering (per agent, per phase, per tool).
- Add persisted view state (last selection, pinned items).

### Phase CM4: Integration + Rollout

- Gate the new renderer behind config flags.
- Provide an adapter registry so multiple UIs can coexist.
- Evaluate integration with the ongoing UI overhaul.

## Risks + Mitigations

- **UI divergence**: keep a stable adapter protocol shared by all renderers.
- **IPC overhead**: batch updates and use binary protocols where needed.
- **Maintenance cost**: limit native renderer scope to “chat mode” first.
- **Cross-platform friction**: prototype on macOS and validate on Windows node.

## Open Questions

- Which renderer candidate best matches the desired UX (terminal vs GPU)?
- Should chat mode be a full screen app or a “panel” inside the main UI?
- Is a web-based renderer acceptable for local workflows?
- What level of replay/scrollback persistence is required?
