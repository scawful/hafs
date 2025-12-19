# Cognitive Protocol Implementation Review (HAFS / oracle-code / halext-org)

This document reviews `Code/hafs/docs/COGNITIVE_PROTOCOL.md` against the *current* implementations in:

- **HAFS terminal client** (`Code/hafs/`)
- **oracle-code** (OpenCode fork, `Code/oracle-code/`)
- **halext-org** suite (`Code/halext-org/`) including:
  - iOS app (`Code/halext-org/ios/`)
  - backend APIs (`Code/halext-org/backend/`)
  - background agent worker (`Code/halext-org/scripts/workers/agent_worker.py`)
  - context sync tooling (`Code/halext-org/scripts/ctx-sync.sh`, `Code/halext-org/scripts/ctx-api-sync.sh`)

It identifies protocol gaps, divergences between implementations, newly added emotion systems, and recommends improvements to the protocol itself (and, where helpful, to the implementations).

## 0. Protocol Documents in This Repo

- **v0.1 (conceptual)**: `Code/hafs/docs/COGNITIVE_PROTOCOL.md`
  - Defines the “Deliberative Context Loop” with a narrative `scratchpad/state.md` template and `memory/fears.json`-driven “concern/anxiety” modeling.
- **v0.2 (schema + cross-impl)**: `Code/hafs/docs/PROTOCOL_SPEC.md`
  - Defines concrete schemas for history/session logging, metacognition, goals, epistemic state, analysis modes, hivemind + council, event bus, etc.
  - Explicitly targets compatibility across **hafs** and **oracle-code**.

In practice, most cross-implementation *interoperability* requirements are in **PROTOCOL_SPEC.md**, while **COGNITIVE_PROTOCOL.md** still describes a v0.1-style “inner monologue” workflow that not all implementations (notably oracle-code and halext-org) actually use.

## 1. Agentic File System (AFS) Layout: What Each Implementation Treats as “The Mind”

### 1.1 Shared Core Directories (Common Across HAFS + oracle-code)

Both HAFS and oracle-code treat `.context/` as the AFS root with these logical directories:

- `memory/` (read_only)
- `knowledge/` (read_only)
- `history/` (read_only for agents; append-only for the system)
- `scratchpad/` (writable)
- `tools/` (executable)

Evidence:
- HAFS config defaults: `Code/hafs/src/hafs/config/schema.py`
- oracle-code AFS dirs: `Code/oracle-code/packages/oracle-code/src/afs/index.ts`

### 1.2 halext-org’s AFS Adds Multi-Tenancy + Extra Mounts

halext-org’s server-side context is multi-tenant and typically lives under:

- `/srv/halext.org/context/<user>/` (user scope)
- `/srv/halext.org/context/global/` (shared scope)

It also includes additional directories beyond the “core 5”, notably:

- `hivemind/` (read_only for agents; system-managed)
- `notes/` (read_only, synced human notes)

Evidence:
- AFS tools policies include `hivemind` + `notes`: `Code/halext-org/tools/agent-tools/afs_tools.py`
- Sync tooling creates/rsyncs these dirs: `Code/halext-org/scripts/ctx-sync.sh`

### 1.3 Key Difference: “Per-Project” vs “Per-User” Context

- **HAFS** and **oracle-code** primarily operate in a **per-project** `.context/` (found/created in the repo/worktree).
- **halext-org** largely treats AFS as a **per-user** identity context, synced across devices (iOS/Web/Mac/Server).

This is a protocol-level fork: v0.2 allows both (see `Code/hafs/docs/PROTOCOL_SPEC.md` Appendix C), but the “how do these coexist without conflict?” part is underspecified and inconsistently implemented.

## 2. Cognitive Protocol State Files: What Exists Where

### 2.1 HAFS (Python TUI) — Implements v0.1 scaffolding + partial v0.2 compatibility

HAFS ensures/scaffolds:

- `.context/scratchpad/state.md` (v0.1 narrative template)
- `.context/scratchpad/deferred.md`
- `.context/scratchpad/metacognition.json` (snake_case; partial)
- `.context/scratchpad/goals.json` (snake_case; partial)
- `.context/memory/fears.json` (trigger/concern/mitigation format)

Evidence:
- Scaffolding logic: `Code/hafs/src/hafs/core/afs/manager.py`
- “Fear” matching + risk block update: `Code/hafs/src/hafs/ui/screens/orchestrator.py`

HAFS also includes *compat helpers* to read oracle-code style camelCase:
- `Code/hafs/src/hafs/core/protocol/metacognition_compat.py`
- `Code/hafs/src/hafs/core/protocol/goals_compat.py`

But HAFS does **not** scaffold or fully manage v0.2 scratchpad files such as:
- `scratchpad/emotions.json`
- `scratchpad/epistemic.json`
- `scratchpad/analysis-triggers.json`
- `scratchpad/grounding.json`

It can *summarize* `epistemic.json` if present, but does not create/maintain it:
- `Code/hafs/src/hafs/core/protocol/prompt_context.py`

### 2.2 oracle-code — Implements the most complete v0.2 feature set (and extends it)

oracle-code maintains a full cognitive suite in `.context/scratchpad/`:

- `metacognition.json` (camelCase, richer than HAFS)
- `goals.json` (camelCase)
- `epistemic.json`
- `emotions.json` (expanded categories beyond v0.2)
- `analysis-triggers.json`
- `grounding.json` (present in practice; referenced by modules)
- `state.md` (structured key-value state + appended cognitive export)

Evidence:
- Prompt injection block building: `Code/oracle-code/packages/oracle-code/src/cognitive/integration.ts`
- State format + cognitive export append: `Code/oracle-code/packages/oracle-code/src/state/index.ts`
- Emotions: `Code/oracle-code/packages/oracle-code/src/cognitive/emotions.ts`
- Epistemic: `Code/oracle-code/packages/oracle-code/src/cognitive/epistemic.ts`
- Metacognition: `Code/oracle-code/packages/oracle-code/src/cognitive/metacognition.ts`
- Hivemind + council: `Code/oracle-code/packages/oracle-code/src/cognitive/hivemind/`

### 2.3 halext-org — Splits “cognition” across file-based AFS, database, and clients

halext-org has three concurrent representations:

1. **File-based AFS** on server (`/srv/halext.org/context/<user>/...`) used by sync scripts and some tools
2. **Database-backed context + hivemind** used for iOS/Web prompt injection and CRUD
3. **Background agent worker prompts** that *intend* to use cognitive context but currently do so incompletely

Evidence:
- File-to-DB importer: `Code/halext-org/backend/app/routers/context.py` (`import_context_from_files`)
- Background agent router (intended flow): `Code/halext-org/backend/app/routers/background_agents.py`
- Background worker (currently TODO for context retrieval): `Code/halext-org/scripts/workers/agent_worker.py`
- iOS prompt context is DB-driven: `Code/halext-org/ios/Cafe/Core/Persistence/Context/ContextModels.swift` (`PromptContext`)

## 3. “New Emotions and Systems Added” vs v0.1

### 3.1 Emotions: v0.1 → v0.2 → oracle-code extensions

**COGNITIVE_PROTOCOL.md (v0.1)** models “anxiety/concern” primarily via:
- `memory/fears.json` triggers
- a per-action confidence score
- a qualitative mitigation strategy

**PROTOCOL_SPEC.md (v0.2)** adds a more explicit `EmotionsState` schema:
- mood (valence + intensity)
- anxiety (level + sources)
- satisfactions + frustrations (as lists)
- mood history

**oracle-code** goes further and implements an “Emotional Valence” system with:
- persistent categories: `fear`, `curiosity`, `satisfaction`, `frustration`, `excitement`, `determination`, `caution`, `relief`
- decay, prune thresholds, mode calibration, and compound emotion interactions

Evidence:
- Expanded emotional model: `Code/oracle-code/packages/oracle-code/src/cognitive/emotions.ts`

**halext-org (iOS + backend)** currently models “emotion-like” long-term entries via **hivemind categories**:
- `fear`, `satisfaction`, `knowledge`, `decision`, `preference`

Evidence:
- iOS models: `Code/halext-org/ios/Cafe/Core/Persistence/Context/ContextModels.swift`

### 3.2 Systems: v0.2 adds (and oracle-code largely implements)

From `Code/hafs/docs/PROTOCOL_SPEC.md`, the v0.2 protocol introduces:

- Immutable **history pipeline** (JSONL)
- **Sessions** + summaries
- **Metacognition** schema
- **Goals** schema
- **Epistemic** schema
- Research-backed **analysis modes** + triggers
- **Hivemind** (cross-session memory)
- **Council** (multi-agent voting / contested→golden promotion)
- **Event bus**

**oracle-code** implements most of these and wires them to tool/message events.

**HAFS** implements:
- history logging primitives (`Code/hafs/src/hafs/core/history/`)
- metacognition/goals scaffolding + ToM marker UI
but does not implement hivemind/council/event bus or the full analysis-mode pipeline.

**halext-org** implements:
- hivemind + council (DB + APIs)
- context sync tooling
but the background worker currently does not actually execute the “Deliberative Context Loop” it documents.

## 4. Cross-Implementation Divergences (Where Compatibility Breaks Today)

### 4.1 `state.md` format divergence (major)

There are at least three incompatible `state.md` formats in play:

1. **HAFS v0.1 narrative template** (headings like “Current Context / Deliberation & Intent”)
   - `Code/hafs/src/hafs/core/afs/manager.py` (`DEFAULT_STATE_TEMPLATE`)

2. **oracle-code structured key-value state**
   - `Code/oracle-code/packages/oracle-code/src/state/index.ts` parses:
     - `## section`
     - `- **key**: value [timestamp]`

3. **halext-org scripts/backends expect different key-value shapes**
   - `Code/halext-org/scripts/ctx-api-sync.sh` parses:
     - `## section`
     - `key: value`
   - `Code/halext-org/backend/app/routers/context.py` expects:
     - `## section`
     - `- key: value`

Impact:
- File-to-DB import and API sync can silently drop or mis-parse state entries.
- HAFS “Deliberative Context Loop” content is not machine-readable by halext-org or oracle-code.

Protocol suggestion:
- Standardize `state.md` as a strict, machine-parseable format *or* move to `state.json` as canonical and treat `state.md` as a rendered view.

### 4.2 JSON key casing (snake_case vs camelCase)

- HAFS historically uses **snake_case** (Pydantic).
- oracle-code uses **camelCase** (Zod).

HAFS added compatibility layers for `metacognition.json` and `goals.json`, but this is incomplete coverage (epistemic/emotions/hivemind are not normalized in HAFS).

Evidence:
- `Code/hafs/src/hafs/core/protocol/metacognition_compat.py`
- `Code/hafs/src/hafs/core/protocol/goals_compat.py`

Protocol suggestion:
- Make a canonical “wire format” per file (recommend: camelCase for JSON, since most client stacks prefer it), include `schema_version`, and provide a formal JSON Schema + migration notes.

### 4.3 “Fears” live in multiple places with different meanings

Today:
- HAFS uses `memory/fears.json` as “trigger conditions → mitigation”.
- oracle-code has `fear` both as an emotion category and as hivemind entries.
- halext-org stores fears as hivemind entries (`hivemind/fears.json`) and sometimes also has `memory/fears.json`.

Protocol suggestion:
- Split fear into:
  - **Tactics**: “hazard triggers → mitigation playbooks” (fits `memory/`)
  - **Learned memories**: “what went wrong / what worked” (fits `hivemind/`)
  - **Session affect**: “current anxiety/confidence” (fits `scratchpad/emotions.json`)

### 4.4 Background worker does not implement the documented loop (halext)

`Code/halext-org/scripts/workers/agent_worker.py` currently:
- builds a prompt with context boundaries
- does **not** fetch user context from the API (explicit TODO)
- does **not** update `scratchpad/state.md`, `metacognition.json`, or hivemind entries

Impact:
- halext background agents are not currently “cognitive protocol compliant” in practice, despite backend docs describing that behavior.

### 4.5 Enforcement vs observability

Across all implementations, most of the protocol is *observed and displayed* rather than *enforced*:
- HAFS provides UI panels and scaffolding, but the actual CLI agents can still act without writing deliberation.
- oracle-code tracks tool/message events and updates cognitive files automatically (strongest enforcement via interception).
- halext background worker currently lacks both enforcement and state writing.

Protocol suggestion:
- Define minimal compliance requirements for “protocol adherence” and an automated check:
  - “No tool execution without prior cognitive snapshot write” (or explicit waiver)
  - “History entry must be appended for every tool call”

## 5. Recommendations to Improve the Protocol (Spec-Level)

### 5.1 Define canonical artifacts and mark everything else as “rendered views”

Recommended canonical sources of truth:
- `scratchpad/state.json` (machine-parseable)
- `scratchpad/metacognition.json`
- `scratchpad/goals.json`
- `scratchpad/emotions.json`
- `scratchpad/epistemic.json`
- `hivemind/*.json` (or a single `hivemind.json` with categories; pick one)
- `history/YYYY-MM-DD.jsonl`

Treat as rendered/optional:
- `scratchpad/state.md` (generated from state.json + cognitive export)
- UI panels, summaries, and dashboards

This removes the current ambiguity where markdown is simultaneously “the mind” and “a UI”.

### 5.2 Add `schema_version` and `producer` metadata to every JSON file

Every JSON protocol file should start with something like:
- `schema_version`: e.g., `"0.2"`
- `producer`: `{ name: "oracle-code", version: "...", instance_id: "..." }`
- `last_updated`

This makes cross-device sync and migrations tractable.

### 5.3 Specify global-vs-project scoping rules (and conflict resolution)

The protocol should explicitly answer:
- Which facts belong in project AFS vs global AFS?
- When both exist, which wins?
- What is the promotion path from project → global (and who is allowed)?

`PROTOCOL_SPEC.md` already sketches this (Appendix C), but implementations differ (halext is “per-user by default”, oracle-code is “per-project by default”).

### 5.4 Standardize sync semantics (device IDs, merges, conflicts)

halext-org has a full sync API concept (`device_id`, `last_sync_at`) in `Code/halext-org/scripts/ctx-api-sync.sh`, while oracle-code has a different notion of global storage and transfer.

Protocol should define:
- required fields: `device_id`, `last_seen_at`, `vector_clock` (optional), conflict strategies
- deterministic merge rules per schema (e.g., “dedupe by key; keep newest updatedAt; preserve audit trail”)

### 5.5 Clarify “read_only means agents shouldn’t write, but the system can”

Both HAFS and oracle-code label `history/` as read_only while their *systems* append to it.
This should be an explicit policy rule in the spec to avoid implementers treating it as immutable at the OS permission layer.

### 5.6 Promote security and prompt-injection handling into the protocol core

halext background worker already adds strong boundary markers and sanitization routines:
- `Code/halext-org/scripts/workers/agent_worker.py` (`sanitize_content`, boundary blocks)

Protocol should define:
- required “untrusted content boundaries”
- required sanitization points (user prompts, file reads, tool outputs)
- how cognitive files themselves should be treated (untrusted vs trusted)

## 6. Implementation Suggestions (Pragmatic Next Steps)

### 6.1 HAFS (Python TUI)

- Add scaffolding for v0.2 scratchpad files (at least empty defaults) so prompt injection is consistent.
- Align `state.md` with a parseable format (or adopt `state.json` as canonical and render markdown).
- Extend compatibility normalization beyond metacognition/goals (epistemic, emotions, and/or hivemind).
- Consider integrating tool interception so HAFS can auto-log history + update metacognition like oracle-code does.

### 6.2 halext background worker + backend

- Implement the missing “fetch context” step in `Code/halext-org/scripts/workers/agent_worker.py` and write back results to AFS + DB.
- Fix `state.md` parsing mismatch between:
  - `Code/halext-org/scripts/ctx-api-sync.sh` writer and
  - `Code/halext-org/backend/app/routers/context.py` parser
  by converging on one canonical format.
- Add bulk endpoints for hivemind where needed to reduce N+1 sync writes.

### 6.3 iOS app

- Decide whether the iOS “prompt injection block” should be `<cognitive_state>` (align with HAFS/oracle-code) or a separate tag, and standardize it.
- If iOS aims to display “agent health”, add read-only views for metacognition/emotions/epistemic when available (even if the iOS client doesn’t compute them).

## 7. Research Artifacts (On-Disk)

The papers referenced by `Code/hafs/docs/PROTOCOL_SPEC.md` exist locally in `Documents/Research/`:

- `Documents/Research/2512.05470v1.pdf` (AFS paper)
- `Documents/Research/2510.04950v1.pdf` (prompt politeness / tone)
- `Documents/Research/2512.08296.pdf` (scaling agent systems)
- `Documents/Research/7799_Quantifying_Human_AI_Syne.pdf` (ToM markers / synergy)
- `Documents/Research/3664646.3665664.pdf` (AutoCommenter)
- `Documents/Research/7525.pdf` (ML code review)
- `Documents/Research/3377816.3381736.pdf` (“Where to comment”)

Protocol suggestion:
- Mount these into `knowledge/` (project or global) as a stable, indexed “common ground” reference set for agents.

