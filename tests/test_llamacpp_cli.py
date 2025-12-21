from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from hafs.cli import app

runner = CliRunner()


def _make_config(
    *,
    base_url: str = "http://example.local:11435/v1",
    model: str = "qwen3-14b",
    timeout_seconds: float = 12.0,
    max_tokens: int = 16,
    temperature: float = 0.25,
    context_size: int = 2048,
    api_key_env: str = "LLAMACPP_API_KEY",
) -> SimpleNamespace:
    return SimpleNamespace(
        llamacpp=SimpleNamespace(
            base_url=base_url,
            host=None,
            port=None,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            context_size=context_size,
            api_key_env=api_key_env,
        )
    )


class StatusBackend:
    instances: list["StatusBackend"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.stopped = False
        self.__class__.instances.append(self)

    async def check_health(self):
        return {
            "status": "online",
            "models": ["alpha", "beta"],
            "base_url": self.kwargs.get("base_url"),
        }

    async def stop(self) -> None:
        self.stopped = True


class ProbeBackend:
    instances: list["ProbeBackend"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []
        self.stopped = False
        self._fail_next = True
        self.__class__.instances.append(self)

    async def generate_one_shot(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if prompt != "Warmup." and self._fail_next:
            self._fail_next = False
            raise RuntimeError("Transient failure")
        return "ok" if prompt == "Warmup." else "success"

    async def stop(self) -> None:
        self.stopped = True


def test_llamacpp_status_json(monkeypatch) -> None:
    StatusBackend.instances.clear()
    cfg = _make_config(model="status-model", temperature=0.33, max_tokens=128)
    monkeypatch.setenv("LLAMACPP_API_KEY", "secret")
    monkeypatch.setattr("hafs.cli.commands.llamacpp.load_config", lambda: cfg)
    monkeypatch.setattr("hafs.cli.commands.llamacpp.LlamaCppBackend", StatusBackend)

    result = runner.invoke(app, ["llamacpp", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "online"
    assert payload["models"] == ["alpha", "beta"]
    assert "latency_ms" in payload

    backend = StatusBackend.instances[-1]
    assert backend.kwargs["base_url"] == cfg.llamacpp.base_url
    assert backend.kwargs["model"] == "status-model"
    assert backend.kwargs["api_key"] == "secret"
    assert backend.kwargs["max_tokens"] == 128
    assert backend.kwargs["temperature"] == 0.33
    assert backend.kwargs["context_size"] == 2048
    assert backend.stopped is True


def test_llamacpp_probe_warmup_retry(monkeypatch) -> None:
    ProbeBackend.instances.clear()
    cfg = _make_config(model="probe-model", max_tokens=16, temperature=0.5)
    monkeypatch.setattr("hafs.cli.commands.llamacpp.load_config", lambda: cfg)
    monkeypatch.setattr("hafs.cli.commands.llamacpp.LlamaCppBackend", ProbeBackend)

    result = runner.invoke(
        app,
        ["llamacpp", "probe", "--warmup", "--retries", "1", "--json", "--prompt", "Ping"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["model"] == "probe-model"
    assert payload["response"] == "success"

    backend = ProbeBackend.instances[-1]
    assert backend.stopped is True
    assert [call["prompt"] for call in backend.calls] == ["Warmup.", "Ping", "Ping"]
    assert backend.calls[0]["temperature"] == 0.0
    assert backend.calls[0]["max_tokens"] == 8
    assert backend.calls[1]["temperature"] == 0.5
    assert backend.calls[1]["max_tokens"] == 16
