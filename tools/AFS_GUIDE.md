# Agentic File System (AFS) Guide

## Philosophy: "Everything is Context"
Traditional AI relies on a "context window" (a text buffer). The AFS extends this window into your actual Operating System. By "mounting" relevant directories into a standardized `.context/` tree, the AI can "walk" your knowledge base without needing to be told where files live every time.

## Directory Structure
Every project using AFS should have a `.context/` directory in its root.

-   **`/memory`**: Long-term storage. Docs, specs, architectural decisions.
    -   *AI Action:* Read before planning.
-   **`/knowledge`**: Reference materials. Read-only source code (e.g., disassembly), logs, external library headers.
    -   *AI Action:* Search for truth/patterns.
-   **`/tools`**: Executable scripts, binaries, or sub-agents.
    -   *AI Action:* Execute to gather state or perform tasks.
-   **`/scratchpad`**: Transient storage. Plans, current reasoning, temporary logs.
    -   *AI Action:* Write thoughts here to "think out loud" and share state with the user.
-   **`/history`**: (Optional) Archival of past scratchpads or logs.

## The `ctx` Tool
Located at `Code/hafs/tools/ctx`, this Python script manages the symlinks.

### Commands
1.  **Initialize a Project:**
    ```bash
    cd ~/MyProject
    ctx init
    ```
    *Creates the `.context` folder structure.*

2.  **Mount a Resource:**
    ```bash
    # Mount local docs
    ctx mount docs memory
    
    # Mount an external reference (e.g., a library in another folder)
    ctx mount ~/Code/ReferenceLibrary knowledge --alias lib_ref
    ```
    *Creates a symlink inside `.context/` pointing to the real file.*

3.  **List Context:**
    ```bash
    ctx list
    ```
    *Shows what the AI can currently "see".*

4.  **Clean Up:**
    ```bash
    ctx clean
    ```
    *Removes the `.context` directory (symlinks only, original files are safe).*

## Configuration Parameters
You can configure AFS behavior via `metadata.json` in the `.context` root (created on init).

```json
{
  "created_at": "2025-12-09",
  "agents": [],
  "policy": {
    "read_only": ["knowledge", "memory"],
    "writable": ["scratchpad"],
    "executable": ["tools"]
  }
}
```
*(Note: Current implementation is basic; future versions can enforce these policies programmatically.)*

## Best Practices
1.  **Don't Duplicate, Mount:** Never copy files into `.context`. Use `ctx mount`. This ensures the AI is always reading the *live* version of your docs/code.
2.  **External References:** Use `/knowledge` for things you shouldn't touch (e.g., `usdasm`, `node_modules`, `third_party`).
3.  **The "Shared Brain":** Encourage the AI to write its plan to `.context/scratchpad/plan.md` before coding. This allows you to review its logic before it destructively edits your source files.
