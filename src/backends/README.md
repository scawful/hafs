# Backends

Chat backend module for AI agent orchestration. This is the canonical location for backends.

## Structure

```
backends/
├── __init__.py          # Main exports and registry initialization
├── base.py              # BaseChatBackend, BackendRegistry, ChatMessage
├── cli/                 # PTY-based CLI backends
│   ├── claude.py        # ClaudeCliBackend
│   ├── gemini.py        # GeminiCliBackend
│   └── pty.py           # PtyWrapper infrastructure
├── api/                 # Direct API backends
│   ├── anthropic.py     # AnthropicBackend
│   ├── openai.py        # OpenAIBackend
│   └── ollama.py        # OllamaBackend
├── oneshot/             # One-shot CLI backends (non-PTY)
│   ├── claude.py        # ClaudeOneShotBackend
│   └── gemini.py        # GeminiOneShotBackend
└── wrappers/            # Backend wrappers
    └── history.py       # HistoryBackend
```

## Usage

```python
from backends import BackendRegistry, ClaudeCliBackend

# Get a registered backend
backend = BackendRegistry.get("claude")

# Or create one directly
backend = ClaudeCliBackend(project_dir=Path.cwd())
await backend.start()
await backend.send_message("Hello!")
async for chunk in backend.stream_response():
    print(chunk, end="")
await backend.stop()
```

## Backward Compatibility

For backward compatibility, `hafs.backends` re-exports everything from this module.
New code should import directly from `backends`.
