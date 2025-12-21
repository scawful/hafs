# Cognitive Protocol v0.3 (Source of Truth)

This document supersedes the narrative v0.1 `COGNITIVE_PROTOCOL.md` and the schema-focused v0.2 `PROTOCOL_SPEC.md` by defining a single, canonical "wire format" for the Agentic File System (AFS) cognitive layer and by aligning implementations across **hafs**, **oracle-code**, and **halext-org**.

## 1. Goals

1. Make JSON the canonical source of truth; Markdown is a rendered view.
2. Standardize schema versioning and producer metadata to enable safe sync/merge.
3. Align emotional and cognitive schemas with the richer oracle-code model.
4. Clarify scoping rules (project vs user/global) and conflict resolution.
5. Define minimum compliance for agents (no action without a snapshot + history log).

## 2. Canonical Artifacts (JSON-first)

AFS root: `.context/`

Canonical (authoritative) files:

- `scratchpad/state.json` – machine-readable shared state (facts/assumptions/decisions/preferences/goals/context).
- `scratchpad/metacognition.json` – strategy, load, spin, help-seeking, flow.
- `scratchpad/goals.json` – goal hierarchy + conflicts + focus stack.
- `scratchpad/emotions.json` – extended emotions (see §3).
- `scratchpad/epistemic.json` – golden/work­ing facts, assumptions, unknowns, contradictions.
- `scratchpad/analysis-triggers.json` – pending/automatic analysis triggers.
- `hivemind/*.json` – cross-session knowledge (fear/satisfaction/knowledge/decision/preference).
- `history/YYYY-MM-DD.jsonl` – append-only history (tool calls, cognitive snapshots, messages).

Rendered/optional views:

- `scratchpad/state.md` – generated from `state.json` + cognitive export.
- UI dashboards/panels (textual/web/mobile).

Each JSON file MUST include:

```json
{
  "schema_version": "0.3",
  "producer": {
    "name": "<app>",
    "version": "<semver>",
    "instance_id": "<optional>"
  },
  "last_updated": "<ISO8601>",
  "...": "payload"
}
```

## 3. Emotion Model (aligned with oracle-code)

Categories: `fear`, `curiosity`, `satisfaction`, `frustration`, `excitement`, `determination`, `caution`, `relief` (+ session `mood`, `anxiety`, `confidence`).  
Recommended fields:

- `session`: `mood`, `anxiety`, `confidence`, `mode`
- Per-category stores: `{ id, trigger, context, intensity, tags, related_files, created_at, updated_at }`
- Decay + prune thresholds per category (see oracle-code `emotions.ts` defaults).
- Interactions (optional): compound effects (e.g., fear mitigated → relief + confidence).

`memory/fears.json` is repurposed for **hazard playbooks** (trigger → mitigation), while long-lived "fear learnings" live in `hivemind/fears.json`.

### 3.1 Emotion Interpretation by Mode

Emotional state serves different purposes depending on the agent's operating mode:

#### Build/Code Mode (Functional Heuristics)

- **Purpose**: Emotions are internal heuristics that guide decision-making
- **Anxiety**: High → more thorough verification, additional tests, smaller commits
- **Confidence**: High → take larger steps, operate more autonomously
- **Expression**: Emotions should NOT be explicitly mentioned in output. They silently influence behavior:
  - High anxiety: Add more safety checks, request confirmation before destructive ops
  - High confidence: Proceed with less hedging, take initiative
  - Frustration: Consider switching strategy, ask for help earlier

#### Plan/Review Mode (Internal Signals)

- **Purpose**: Emotions signal when to escalate or change approach
- **Expression**: May reference emotional state when explaining decisions:
  - "Given the complexity, I'm proceeding carefully..."
  - "This matches a known pattern, so I'm confident in..."

#### Chat Mode (Tone Adjustment)

- **Purpose**: Emotions affect communication style without explicit expression
- **Expression**: Tone adjusts implicitly, NOT explicitly stated
  - High anxiety → More hedging language ("I believe...", "It seems...")
  - High confidence → More direct language ("The solution is...", "Do X")
  - Frustration → Acknowledge difficulty without dwelling ("This is tricky, but...")
  - Curiosity → Show engagement ("That's interesting because...")
- **Anti-pattern**: Never say "I feel anxious about this" or "My confidence is 85%"
- **Good pattern**: Naturally vary certainty language based on internal state

#### Implementation Reference

See `oracle-code/src/cognitive/emotions.ts:getExpressionStyle()` for the canonical implementation:

- Returns `hedgingLevel`, `directness`, `warmth` based on mode and emotional state
- Build/plan modes return neutral expression (behavior-only influence)
- Chat mode maps anxiety→hedging, confidence→directness, mood→warmth

## 4. State Format

- Canonical: `state.json`
- Rendered: `state.md` generated from JSON with bullet format:
  - Sections: `facts`, `assumptions`, `decisions`, `uncertainties`, `goals`, `context`
  - Entry: `- **<key>**: <value> [YYYY-MM-DD HH:MM]`

## 5. Scoping and Merge Rules

- **Project scope**: `.context/` in repo/worktree (default for coding assistants).
- **User scope**: `/srv/halext.org/context/<user>/` (halext server) or `~/.context/<user>/`.
- **Global scope**: `/srv/halext.org/context/global/` or `~/.context/global/`.

Merge policy (per file):

- Deduplicate by `id`/`key` where applicable; newest `updated_at` wins.
- Record `producer` and `last_updated` for conflict auditing.
- Hivemind promotion path: project → user → global requires explicit approval or council vote (when available).

## 6. Compliance Requirements (Minimum)

Before any tool execution:

1. Read canonical cognitive files (`state.json`, `metacognition.json`, `goals.json`, `emotions.json`, `epistemic.json` if present, `hivemind/*` if allowed).
2. Write a snapshot to `state.md` (rendered) and append a cognitive state entry to `history/*.jsonl` (operation type: `cognitive_state`).
3. If fears/hazards are matched, include mitigation in the snapshot.

After execution:

1. Append tool result to history (with duration, success flag, files touched).
2. Update metacognition (spin/load/strategy) and emotions (anxiety/confidence) from outcomes.
3. Optionally promote learnings to hivemind (manual or council-mediated).

## 7. Implementation Notes per Client

- **hafs (Python TUI)**: scaffold all canonical files with v0.3 metadata; prefer reading JSON and treating `state.md` as a view. Add compatibility readers for camelCase where needed (meta/goals already present; extend to emotions/epistemic if desired).
- **oracle-code**: already closest to v0.2/extended emotions. Add `schema_version/producer` metadata on write to become fully v0.3-compliant.
- **halext-org**:
  - Background worker must fetch context (API) and write snapshots + history.
  - Unify `state.md` parsing/writing with the canonical rendered format.
  - Keep fears playbooks in `memory/` and long-lived fears in `hivemind/`.

## 8. File Policy Table (v0.3)

| Path                         | Role                    | Policy        |
| ---------------------------- | ----------------------- | ------------- |
| `.context/memory/`           | Long-term docs, hazards | read_only\*   |
| `.context/knowledge/`        | Reference               | read_only\*   |
| `.context/history/`          | Append-only log         | system_append |
| `.context/scratchpad/`       | Working memory          | writable      |
| `.context/tools/`            | Procedural memory       | executable    |
| `.context/hivemind/`         | Cross-session learnings | read_only\*   |
| `.context/notes/` (optional) | Human notes             | read_only\*   |

`read_only*` means agents should not write, but the system/backends may append/update according to the protocol.

## 9. Migration Guidance

1. Add `schema_version`/`producer` on next write of every cognitive JSON.
2. Introduce `state.json` as canonical; regenerate `state.md` from it.
3. Align fears: move "playbooks" to `memory/fears.json`; keep long-lived memories in `hivemind/fears.json`; keep session anxiety in `emotions.json`.
4. If multiple scopes exist, merge by `updated_at`, log conflicts, and prefer user/global only when project data is missing.
5. Standardize sync payloads to include `device_id`, `last_sync_at`, and per-file checksums or vector clocks if available.
