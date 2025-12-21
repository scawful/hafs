# HAFS Improvement Opportunities & Feature Ideas

This document outlines areas for improvement, known issues, and potential new features for the HAFS project.

---

## Related Plans

- `RESEARCH_ALIGNMENT_PLAN.md` for research-driven analysis upgrades
- `CHAT_MODE_RENDERER_PLAN.md` for chat mode + renderer feasibility

## Roadmap (Aligned Phases)

**Phase 0 (Foundation) - Complete**
- Unified config loader + legacy mapping
- Orchestrator bridge to UnifiedOrchestrator v2
- Cognitive layer aligned to v0.3 artifacts

**Phase 1 (History + Recall) - Complete**
- History logging wired into coordinator/swarm
- Entry embeddings + session summaries with semantic search
- CLI/TUI history search support

**Phase 2 (Tooling + Sandboxes) - Complete**
- Tool execution routed through `ToolRunner` + `ExecutionPolicy` across background agents
- AFS policy guard rails (metadata `policy.executable` gating for write/build/test tools)
- Execution modes (`read_only`, `build_only`, `infra_ops`) with env override
- Expanded tool catalog for build/test/infra ops + safer git operations

**Phase 3 (Orchestration Strategy) - Complete**
- Consolidate SwarmCouncil + AgentCoordinator lifecycles (single orchestration entrypoint)
  - Unified entrypoint (`hafs orchestrate run`) + UI `/orchestrate` reuses coordinator lanes
- Persona + skills registry (role -> tools -> constraints -> goals)
- Structured pipelines: plan → execute → verify → summarize
- Agent tool profiles by persona + execution mode
- External provider interfaces formalized (IntegrationPlugin + adapter protocols)

**Phase 4 (Knowledge Expansion) - Complete**
- Shared batch embedding manager with checkpointing across KBs
- Embedding service syncs from ProjectRegistry and indexes cross-repo sources
- Knowledge graph enrichment from disassembly KB routines + symbol access

**Phase 5 (Multi-Node Autonomy) - In Progress**
- Expanded node registry + CLI for multi-role nodes (compute/server/mobile)
- AFS sync profiles + runner + status logging (project/user/global scopes)
- Observability + daemon support for sync checks + infrastructure UI surface
- Health + drift monitoring + autonomous recovery workflows (next)

## Code Quality Improvements

### 1. Error Handling

**Current State**: Many try/except blocks silently catch all exceptions.

**Improvements**:
- Add specific exception types for different failure modes
- Create custom exception hierarchy (`HafsError`, `BackendError`, `ConfigError`, etc.)
- Log errors with context for debugging
- Show user-friendly error messages in TUI

```python
# Example improvement
class HafsError(Exception):
    """Base exception for HAFS."""
    pass

class BackendConnectionError(HafsError):
    """Failed to connect to AI backend."""
    pass

class AgentRegistrationError(HafsError):
    """Failed to register agent."""
    pass
```

### 2. Type Safety

**Current State**: Some areas use `Any` types or lack type hints.

**Improvements**:
- Add complete type hints throughout codebase
- Use `TypedDict` for configuration dictionaries
- Add `mypy` to CI with strict mode
- Use `Protocol` classes for duck-typed interfaces

### 3. Testing

**Current State**: Minimal test coverage.

**Improvements**:
- Add unit tests for core modules (parsers, router, coordinator)
- Add integration tests for backend communication
- Add TUI tests using Textual's testing framework
- Add property-based tests with Hypothesis
- Target 80%+ code coverage

### 4. Configuration Validation

**Current State**: Pydantic models validate structure but not semantics.

**Improvements**:
- Validate paths exist and are accessible
- Validate backend commands are executable
- Warn about conflicting settings
- Add configuration migration for version updates

### 5. Logging

**Current State**: Limited logging, mostly debug-level.

**Improvements**:
- Structured logging with context
- Log rotation and retention policies
- Separate log levels for different components
- Performance timing logs for optimization

---

## UI/UX Improvements

### 1. Visual Design

**Current Issues**:
- Some buttons don't show text clearly
- Modal dialogs could be more polished
- Inconsistent spacing in some areas

**Improvements**:
- Consistent padding/margins throughout
- Better contrast for text on colored backgrounds
- Loading states with progress indicators
- Smooth transitions between screens
- Custom icons for file types and actions

### 2. Keyboard Navigation

**Current Issues**:
- Vim mode conflicts with input fields (recently fixed)
- Some screens lack keyboard shortcuts

**Improvements**:
- Context-aware keybinding help
- Customizable keybindings in config
- Vim-style text objects in inputs
- Quick-jump shortcuts (go to screen by number)

### 3. Accessibility

**Improvements**:
- High contrast theme option
- Screen reader support
- Configurable font sizes
- Color blind-friendly palette option

### 4. Responsive Layout

**Current Issues**:
- Fixed widths may not work on small terminals

**Improvements**:
- Minimum terminal size detection
- Responsive sidebar that collapses
- Adaptive panel sizes based on content
- Mobile-friendly layout for smaller terminals

### 5. Search & Filtering

**Current State**: Basic fuzzy search implemented.

**Improvements**:
- Search across all logs simultaneously
- Search within file contents
- Advanced filters (date range, token count, etc.)
- Search history and saved searches
- Regex search option

### 6. Status Indicators

**Improvements**:
- Real-time backend connection status
- Agent activity indicators (typing, thinking)
- Token usage meters
- Rate limit warnings
- Network latency display

---

## Feature Ideas

### 1. Session Management

**Features**:
- Save and restore chat sessions
- Export conversations to markdown/JSON
- Session branching (fork from any point)
- Session templates for common workflows
- Share sessions between users

### 2. Context Enhancement

**Features**:
- Automatic context summarization
- Context window usage visualization
- Smart context pruning suggestions
- File change tracking in context
- Git integration for diffs in context

### 3. Agent Improvements

**Features**:
- Custom agent creation wizard
- Agent templates (frontend, backend, devops, etc.)
- Agent personality customization
- Agent-to-agent direct communication
- Agent memory persistence across sessions
- Agent skill learning from corrections

### 4. Workflow Automation

**Features**:
- Macro recording and playback
- Scheduled tasks (nightly summaries, etc.)
- Webhook integration for triggers
- Pipeline definitions (agent chains)
- Conditional routing rules

### 5. Collaboration Features

**Features**:
- Multi-user sessions (pair programming)
- Agent response voting/rating
- Shared team configurations
- Activity feed for team awareness
- Comments on log entries

### 6. Integration Ecosystem

**Features**:
- VS Code extension
- JetBrains plugin
- GitHub/GitLab integration
- Jira/Linear task sync
- Slack/Discord notifications
- API for external tools

### 7. Analytics & Insights

**Features**:
- Token usage dashboard
- Cost estimation and tracking
- Response quality metrics
- Agent performance comparison
- Usage patterns and recommendations

### 8. Advanced Chat Features

**Features**:
- Image/file attachment support
- Code execution sandbox
- Rich markdown rendering
- Syntax highlighting in chat
- Inline code editing
- Diff viewer for code changes

### 9. Project Templates

**Features**:
- Pre-configured AFS structures
- Language-specific templates (Python, JS, Rust, etc.)
- Framework templates (React, Django, etc.)
- Best practices included in templates

### 10. Offline Mode

**Features**:
- Local model support (Ollama, llama.cpp)
- Offline log browsing
- Queue messages for later sending
- Local caching of responses

---

## Performance Improvements

### 1. Startup Time

**Current Issues**:
- Initial load can be slow with many projects

**Improvements**:
- Lazy load parsers on first use
- Cache project discovery results
- Defer non-critical initialization
- Profile and optimize hot paths

### 2. Memory Usage

**Improvements**:
- Limit cached sessions in memory
- Stream large files instead of loading fully
- Garbage collect unused widgets
- Use weak references where appropriate

### 3. Responsiveness

**Improvements**:
- Move all I/O to background workers
- Debounce rapid input events
- Virtual scrolling for long lists
- Progressive rendering for large content

---

## Security Improvements

### 1. Credential Management

**Improvements**:
- Secure storage for API keys
- Support for credential managers (1Password, etc.)
- Audit log for sensitive operations
- Token rotation reminders

### 2. Sandboxing

**Improvements**:
- Sandbox agent file operations
- Restrict network access per agent
- Audit trail for all file changes
- Confirmation for destructive operations

### 3. Privacy

**Improvements**:
- Local-only mode option
- Configurable telemetry (opt-in)
- PII detection and warnings
- Data retention policies

---

## Documentation Improvements

### 1. User Documentation

**Needs**:
- Getting started guide
- Configuration reference
- Keybinding cheat sheet
- Troubleshooting guide
- Video tutorials

### 2. Developer Documentation

**Needs**:
- Contributing guide
- Architecture deep-dives
- Plugin development guide
- API documentation
- Code style guide

### 3. In-App Help

**Improvements**:
- Contextual help tooltips
- Interactive tutorials
- Example commands in palette
- Link to docs from error messages

---

## Infrastructure Improvements

### 1. Build & Release

**Improvements**:
- Automated releases with changelog
- Binary distributions (PyInstaller, etc.)
- Homebrew formula
- Docker image
- Version update notifications

### 2. CI/CD

**Improvements**:
- Automated testing on PR
- Code coverage reporting
- Dependency vulnerability scanning
- Performance regression testing
- Cross-platform testing

### 3. Monitoring

**Improvements**:
- Crash reporting (opt-in)
- Usage analytics (opt-in)
- Performance metrics collection
- Feature flag system

---

## Priority Ranking

### High Priority (Fix Soon)
1. Improve error handling and user feedback
2. Add comprehensive test suite
3. Fix remaining UI/UX issues
4. Complete documentation

### Medium Priority (Next Features)
1. Session save/restore
2. Better search and filtering
3. Git integration
4. VS Code extension

### Low Priority (Future)
1. Multi-user collaboration
2. Analytics dashboard
3. Offline mode
4. Mobile support

---

## Contributing

If you'd like to work on any of these improvements:

1. Check existing issues on GitHub
2. Open an issue to discuss the approach
3. Submit a PR with tests
4. Update documentation as needed

See the contributing guide for more details.
