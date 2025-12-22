# Cloud Testing with Halext AI Nodes

This document explains how to set up and use the Halext AI node infrastructure for running tests in the cloud and loading custom models (GGUF/Modelfiles).

## 1. GitHub Actions Integration

The repository is configured with an AI-aware CI pipeline in `.github/workflows/ci-ai.yml`.

- **Service Container**: Runs a local Ollama instance on port 11434.
- **Model Pre-loading**: Automatically pulls `qwen2.5-coder:1.5b` for fast, reproducible tests.
- **Environment Variables**:
    - `HAFS_ENABLE_OLLAMA=true`: Enables Ollama backend discovery.
    - `OLLAMA_HOST=http://localhost:11434`: Points to the CI service container.

## 2. Halext AI Nodes

Halext AI nodes provide remote inference capabilities with authentication.

### Configuration

Add a Halext node to your `hafs.toml` or environment:

```toml
[[nodes]]
name = "halext-cloud"
node_type = "halext"
gateway_url = "https://api.halext.org/v1"
api_token = "YOUR_TOKEN"
org_id = "YOUR_ORG_ID"
```

### Routing

The `UnifiedOrchestrator` will automatically detect `HalextNode` instances. You can force routing to the Halext provider:

```python
orchestrator = UnifiedOrchestrator()
result = await orchestrator.generate(
    prompt="...",
    provider=Provider.HALEXT
)
```

## 3. Custom Model Loading

You can scaffold custom model loading (GGUF or Modelfiles) onto specific nodes:

```python
await orchestrator.load_custom_model("path/to/my-model.gguf", node_name="halext-cloud")
```

> [!NOTE]
> Currently, `load_custom_model` is a scaffold. Future updates will implement the logic for uploading and registering models via the Halext gateway or directly to Ollama nodes.

## 4. LSP Integration

The `hafs-lsp` currently defaults to a local Ollama instance. To point it to a remote Halext node, update `lsp.toml`:

```toml
[server]
enabled = true
model = "halext:qwen3:14b-reasoning"
```

The LSP will use the `UnifiedOrchestrator` (v2) to resolve this model and route to the appropriate cloud node.
