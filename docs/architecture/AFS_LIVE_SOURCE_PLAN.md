# AFS Live Source Plan

## Context
The viz app currently loads AFS training telemetry from local JSON snapshots
(`~/.context/training`). We want to support live data streams from AFS nodes
(e.g., local AI nodes or cloud models) while keeping snapshot files as a safe,
offline fallback. Live connectivity is deferred until the server side is ready.

## Goals
- Support both file snapshots and live sources behind a single data interface.
- Preserve current snapshot behavior and file locations.
- Improve data freshness/staleness visibility in the UI.
- Fail safely: on live source errors, fall back to last good snapshot.
- Keep JSON parsing robust against partial writes.

## Non-Goals (Current Phase)
- No live AFS transport implementation yet (server is not ready).
- No breaking changes to existing JSON formats.

## Proposed Architecture

### DataSource Interface (C++)
- `Refresh()` updates an internal cache and load diagnostics.
- `GetSnapshot()` returns a full snapshot struct (see schema doc).
- `GetStatus()` returns health, error counts, and staleness.

### Implementations
- `FileDataSource` (now): reads JSON snapshots from a configured path.
  - Uses robust JSON parsing with guardrails.
  - Holds the last good snapshot if a file is invalid or partially written.
- `LiveDataSource` (later): pulls from a live AFS endpoint.
  - Transport: HTTP JSON or gRPC (to be decided).
  - Timeouts and retries with exponential backoff.
  - Partial updates merge into a cached snapshot.

### Snapshot Cache
- Snapshot is immutable once published to the UI per frame.
- Track `generated_at` and `sequence` to detect out-of-order updates.

## UI Considerations
- Always show source health: `sources ok/total`, last update, errors.
- Mark simulated or proxy metrics when no data is present.
- Show staleness warnings when data is older than a threshold.

## Telemetry Emission (Foundation)
- Emitters should write JSON snapshots atomically (write temp + rename).
- Include metadata: `generated_at`, `sequence`, `source`.
- Do not block training; telemetry should be best-effort.

## Rollout Steps
1. Finalize snapshot schema (see `docs/architecture/TRAINING_SNAPSHOT_SCHEMA.md`).
2. Implement and ship file-backed DataSource with diagnostics and UI health.
3. Add emitter hooks to produce snapshots from training/AFS services.
4. Add live source implementation when server is ready.
5. Gate live source via config/env flag; keep snapshots as fallback.

## Risks and Mitigations
- Partial JSON writes: guard with parse checks and last-good caching.
- Schema drift: maintain schema doc + lightweight validation.
- Mixed data freshness: include timestamps and staleness indicators.
