# Knowledge Base Next Steps

This file tracks follow-ups for the knowledge base and test harness work.

## 1) Async test harness

Problem:
- Several KB and integration scripts define async tests without an async test
  runner, causing pytest to fail with "async def functions are not natively supported."

Options:
- Adopt `pytest-asyncio` and mark async tests with `@pytest.mark.asyncio`.
- Or wrap async tests with `asyncio.run(...)` and keep them sync.
- If using anyio, mark with `@pytest.mark.anyio` and align fixtures.

Acceptance:
- `pytest scripts/test_alttp_unified_kb.py` runs without async errors.
- Async tests are consistently marked and documented.

## 2) Integration test markers

Problem:
- KB scripts hit external resources (embeddings, Ollama, file system) and
  shouldn't block default `pytest`.

Plan:
- Add `@pytest.mark.integration` to heavy/externally dependent scripts.
- Document `pytest -m "not integration"` as the default CI/local path.
- Optionally gate with `HAFS_RUN_INTEGRATION=1`.

Acceptance:
- Default pytest run excludes integration scripts.
- Integration run is explicit and reproducible.

## 3) Legacy import cleanup

Problem:
- KB scripts still use `hafs.agents.*` legacy shims.

Plan:
- Update scripts to import from `agents.knowledge.*` directly.
- Keep `hafs.agents.alttp_unified_kb` as a compatibility layer (already in place).
- Update any docs/examples referencing legacy paths.

Acceptance:
- KB scripts import from `agents.knowledge.*` only.
- Legacy shims remain for compatibility, but docs use the new paths.
