# AFS Architecture & Contract

**Agentic File System (AFS)** is a standard for organizing AI context on disk.

## Core Principle: The Filesystem is the Database
There is no central server process that mediates access. The `.context` directory is the single source of truth.

### The "Split-Stack" Implementation
Because the user's ecosystem uses both Python (CLI/TUI) and TypeScript (VS Code/Web), there are **two** implementations of the AFS protocol.

1.  **Python (`hafs` repo):**
    *   **Role:** Administrative CLI (`afs init`), TUI explorer, and Python agent interface.
    *   **Logic:** `src/hafs/core.py` (Manager).
    *   **Use Case:** Interactive sessions, shell scripts, Python-based swarms.

2.  **TypeScript (`@halext/afs` package):**
    *   **Role:** IDE integration, Context Evaluation, and VS Code extension support.
    *   **Logic:** `src/index.ts` (AFS Class).
    *   **Use Case:** `opencode` integration, React apps, Language Servers.

## The Contract (Directory Structure)
Both implementations **MUST** adhere to this directory layout. Any change here breaks the other tool.

```
.context/
├── metadata.json       # Versioning and global agent settings
├── memory/             # Long-term storage (Docs, Specifications)
│   └── active/         # Currently relevant context
├── knowledge/          # Read-only reference (Libraries, Logs)
├── tools/              # Executable scripts for agents
├── scratchpad/         # Transient workspace (Plans, Drafts)
└── history/            # Interaction logs
```

## Consistency Rules
1.  **Locking:** Since there is no daemon, concurrent writes are possible. Agents should rely on file-level locking or append-only logs (`history/`) where possible.
2.  **Metadata:** `metadata.json` MUST contain a `version` field. If `hafs` encounters a version higher than it supports, it must warn the user.
3.  **Paths:** All paths in `metadata.json` or cross-references must be relative to the project root, not absolute system paths.

## Development Workflow
When adding a feature (e.g., "Vector Search"):
1.  Define the folder structure in this document.
2.  Implement in `hafs` (Python) first to verify CLI usage.
3.  Port the interface to `@halext/afs` (TypeScript) for IDE support.
