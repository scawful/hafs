# HAFS Research Alignment Plan

This plan captures next steps based on the research audit in `~/Documents/Research`
and the current gaps in HAFS (synergy/ToM analysis, scaling metrics, review/doc
analyzers, and rationale capture). It is intended to complement
`docs/IMPROVEMENTS.md` rather than replace it.

## Goals

- Align HAFS analysis modes with the research themes on multi-agent scaling,
  human-AI synergy, and review/documentation quality.
- Make analysis outputs actionable in orchestration (routing, tool selection,
  and autonomous follow-ups).
- Persist analysis results in history for retrieval, comparison, and learning.

## Non-Goals

- Major UI redesigns or reworking the AFS policy model.
- Re-indexing all knowledge bases unless explicitly required by analysis.

## Inputs

- Research topics: agent scaling, synergy/ToM, review quality, documentation
  placement, developer productivity, error repair, and prompt behavior.
- Existing HAFS components: analysis registry, `HistoryLogger`, orchestrator
  pipeline, and AFS rationale scaffolding.

## Phase RA0: Spec + Schema Alignment

- Define analysis schemas for:
  - `synergy_tom`: interaction signals, initiative balance, mismatch flags.
  - `scaling_metrics`: agent coordination cost, handoff latency, saturation.
  - `review_quality`: severity taxonomy and evidence scoring.
  - `doc_quality`: coverage, locality, and drift heuristics.
- Add config toggles + versioning for each analysis mode.
- Create small fixtures in `tests/` to validate schema stability.

## Phase RA1: Analyzer Implementations

- Implement a `SynergyToMAnalyzer` that uses conversation + tool traces to
  compute deltas instead of static placeholders.
- Implement a `ScalingMetricsAnalyzer` aligned to the scaling research
  vocabulary (coordination overhead, parallelism, merge cost).
- Implement a `ReviewQualityAnalyzer` that tags issues by severity and points
  to evidence artifacts.
- Implement a `DocumentationQualityAnalyzer` that measures doc/code locality
  and drift against runtime changes.
- Add `RationaleCapture` hooks to AFS to automatically capture intent from
  diffs, issue IDs, and commit messages (in addition to manual inputs).

## Phase RA2: Integration + Surface Area

- Register the new analyzers in `AnalysisModeRegistry`.
- Route analysis outputs into `context_builder` for downstream routing.
- Add CLI entry points (e.g., `hafs analysis run --mode synergy_tom`).
- Surface results in the TUI and web dashboard (summary + drill-down links).

## Phase RA3: Feedback Loops + Autonomy

- Embed analysis summaries into history for retrieval and comparisons.
- Use analysis signals to:
  - route tasks to specific personas or tools,
  - trigger remediation tasks (follow-up reviews, docs updates),
  - enforce quality gates during orchestrated runs.
- Add scheduled runs for recurring evaluation.

## Phase RA4: Multi-Node + External Systems

- Extend analysis to multi-node deployments (macOS, halext-server, Windows node,
  iOS app) via AFS sync and node metadata.
- Add node-specific tool profiles and safety policies.
- Aggregate per-node analysis into a global health/quality report.

## Implementation Map (Proposed)

- `src/hafs/core/analysis/` for analyzer classes and registries.
- `src/hafs/agents/context_builder.py` for routing analysis outputs.
- `src/hafs/core/afs/` for automated rationale capture.
- `src/hafs/services/history_logger.py` for embedding summaries.
- `src/hafs/ui/` for surfacing results.

## Success Criteria

- Analysis modes produce non-placeholder outputs with documented heuristics.
- Orchestration consumes analysis outputs in routing decisions.
- Analysis summaries are searchable via history embeddings.
- Clear per-node and global quality reports exist for operator review.

## Risks + Mitigations

- **Signal noise**: start with conservative heuristics and add thresholds.
- **Data drift**: version analysis schemas and log toolchain versions.
- **Privacy**: keep sensitive analysis outputs in local AFS by default.

## Open Questions

- Which analysis outputs should be exposed by default vs. opt-in?
- What evaluation datasets should be considered canonical for regression tests?
