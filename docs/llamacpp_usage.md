# llama.cpp Advanced Model Control Documentation

This documentation explains how to use the advanced model control parameters introduced for the `llama.cpp` backend in `hafs`.

## CLI Usage

The `hafs llamacpp probe` command now supports several new options to control model generation fine-tuning.

### Basic Sampling
- `--temperature`: Sampling temperature (default: 0.7).
- `--top-p`: Nucleus sampling top-p (default: 0.9).
- `--top-k`: Top-k sampling (default: 40).
- `--min-p`: Minimum probability threshold (default: 0.05).

### Penalties
- `--repeat-penalty`: Penalty for repeating tokens (default: 1.1).
- `--presence-penalty`: Penalty for token presence.
- `--frequency-penalty`: Penalty for token frequency.

### Mirostat Control
Mirostat is an algorithm that maintains the perplexity of the generated text within a desired range.
- `--mirostat`: Mirostat mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
- `--mirostat-tau`: Target entropy (default: 5.0).
- `--mirostat-eta`: Learning rate (default: 0.1).

### Stop Sequences
- `--stop`: One or more stop sequences. If the model generates any of these, it will stop immediately.
  ```bash
  hafs llamacpp probe --prompt "Write a list" --stop "\n" --stop "5."
  ```

## Configuration

You can also configure these parameters in your `hafs.toml` file under the `[llamacpp]` section.

```toml
[llamacpp]
enabled = true
base_url = "http://localhost:11435/v1"
model = "qwen3-14b"
temperature = 0.8
top_p = 0.95
repeat_penalty = 1.2
mirostat = 2
```

## Orchestrator Integration

If you are using the `LocalAIOrchestrator` programmatically, you can pass these parameters in the `InferenceRequest`:

```python
from hafs.services.local_ai_orchestrator import InferenceRequest, RequestPriority

request = InferenceRequest(
    id="custom_request",
    priority=RequestPriority.INTERACTIVE,
    prompt="Your prompt here",
    top_p=0.8,
    repeat_penalty=1.05
)
```

## REST API Details

The `LlamaCppBackend` sends these parameters directly to the `/chat/completions` endpoint of your `llama.cpp` (or compatible) server in the JSON payload. Ensure your server version supports these OpenAl-compatible parameters.
