# Training Preparation & Future-Proofing Guide

**Created**: 2025-12-21
**System**: hAFS Training Pipeline
**Hardware**: medical-mechanica (RTX 5060 Ti 16GB)

---

## 1. Current Hardware Feasibility

### medical-mechanica Specs
- **GPU**: NVIDIA RTX 5060 Ti 16GB VRAM
- **OS**: Windows (Ollama + WSL2)
- **Network**: Tailscale VPN at 100.100.100.20:11434
- **Purpose**: Local inference + fine-tuning

### Training Feasibility Analysis

#### ✓ **FEASIBLE: Qwen2.5-Coder:14B with 4-bit LoRA**
```toml
[training.qwen_14b]
model = "qwen2.5-coder:14b"
quantization = "4-bit"
adapter = "LoRA"
rank = 16
batch_size = 2
gradient_accumulation = 4
effective_batch = 8  # 2 * 4

# VRAM Estimate:
# - Model: ~8GB (4-bit quantized)
# - LoRA adapters: ~2GB
# - Optimizer states: ~3GB
# - Batch processing: ~2GB
# TOTAL: ~15GB (fits in 16GB with 1GB headroom)
```

**Expected Performance**:
- Training speed: ~2-3 tokens/sec
- Time for 24K samples (3 epochs): **8-12 hours**
- Time for 7K samples (3 epochs): **3-5 hours**
- Total for both models: **12-16 hours** ✓

#### ⚠️ **MARGINAL: Qwen2.5-Coder:32B with 4-bit LoRA**
```toml
[training.qwen_32b]
model = "qwen2.5-coder:32b"
quantization = "4-bit"
adapter = "LoRA"
rank = 8  # Reduced from 16
batch_size = 1
gradient_accumulation = 8
effective_batch = 8

# VRAM Estimate:
# - Model: ~16GB (4-bit)
# - LoRA adapters: ~1.5GB (rank=8)
# TOTAL: ~17.5GB (EXCEEDS 16GB)
```

**Verdict**: Need to use Qwen2.5-Coder:14B, not 32B.

#### ❌ **NOT FEASIBLE: Full fine-tuning**
- Requires >48GB VRAM even with quantization
- Alternative: Use Unsloth Cloud or RunPod

### Recommended Training Strategy

**Option 1: Local Training (medical-mechanica)**
```bash
# Train both agents sequentially on medical-mechanica
# Advantages: Free, full control
# Disadvantages: Slower, limited to 14B model

# Agent 1: ALTTP ASM (24K samples)
python -m agents.training.scripts.train_agent \
    --model qwen2.5-coder:14b \
    --dataset alttp_asm_24k \
    --output alttp_asm_agent \
    --hours 8-12

# Agent 2: YAZE Tools (7K samples)
python -m agents.training.scripts.train_agent \
    --model qwen2.5-coder:14b \
    --dataset yaze_tools_7k \
    --output yaze_tool_agent \
    --hours 3-5

# Total: 12-16 hours
```

**Option 2: Hybrid (Gemini generation + RunPod training)**
```bash
# Use RunPod with 24GB+ GPU for larger models
# Qwen2.5-Coder:32B with 4-bit LoRA
# Cost: ~$0.50-1.00/hr
# Time: 6-8 hours for both agents
# Total cost: ~$4-8
```

**Recommendation**: Start with **Option 1** (local). If quality is insufficient, upgrade to Option 2 for 32B model.

---

## 2. OpenRouter Integration

### Why OpenRouter?
- **Access to 100+ models** via single API
- **Automatic fallbacks** if quota exceeded
- **Cost optimization** (some models cheaper than direct APIs)
- **Latest models** (GPT-5, Claude Opus 4.5, Gemini 3, DeepSeek R1)

### Implementation Plan

**Step 1: Add OpenRouter Provider**

```python
# src/hafs/backends/openrouter.py
class OpenRouterBackend(BaseChatBackend):
    """OpenRouter backend with 100+ models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-5.2",
        site_url: str = "https://hafs.dev",
        app_name: str = "hAFS",
    ):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._model = model
        self._site_url = site_url
        self._app_name = app_name
        self._base_url = "https://openrouter.ai/api/v1"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._site_url,
            "X-Title": self._app_name,
        }

    async def generate_one_shot(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate via OpenRouter API (OpenAI-compatible)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature or 0.7,
                },
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
```

**Step 2: Add to models.toml**

```toml
[providers.openrouter]
enabled = true
api_key_env = "OPENROUTER_API_KEY"
site_url = "https://hafs.dev"
app_name = "hAFS"

[providers.openrouter.models]
fast = "google/gemini-3-flash-preview"
coding = "deepseek/deepseek-r1"  # Best code model on OpenRouter
reasoning = "openai/o3"
cheap = "meta-llama/llama-3.3-70b"  # Free tier
fallback = "anthropic/claude-3-haiku"

[routing.coding]
providers = ["openai", "openrouter", "gemini", "anthropic", "ollama"]
fallback_order = [
    "gpt-5.2",
    "deepseek/deepseek-r1",  # OpenRouter fallback
    "gemini-3-flash-preview",
    "claude-sonnet-4-5-20250929",
    "qwen2.5-coder:14b"
]
```

**Step 3: Quota Exhaustion Handling**

```python
# src/hafs/core/quota.py (enhancement)
class QuotaManager:
    async def handle_quota_exceeded(
        self,
        provider: Provider,
        tier: TaskTier,
    ) -> tuple[Provider, str]:
        """Handle quota exceeded by rotating to OpenRouter."""
        logger.warning(f"{provider} quota exceeded, rotating to OpenRouter")

        # Try OpenRouter as universal fallback
        if provider != Provider.OPENROUTER:
            openrouter_model = self._get_openrouter_equivalent(provider, tier)
            return (Provider.OPENROUTER, openrouter_model)

        # If OpenRouter also exhausted, use local
        return (Provider.OLLAMA, "qwen2.5-coder:14b")

    def _get_openrouter_equivalent(
        self, provider: Provider, tier: TaskTier
    ) -> str:
        """Map provider to OpenRouter equivalent."""
        mapping = {
            Provider.OPENAI: {
                TaskTier.FAST: "openai/gpt-3.5-turbo",
                TaskTier.CODING: "openai/gpt-5.2",
                TaskTier.REASONING: "openai/o3",
            },
            Provider.ANTHROPIC: {
                TaskTier.FAST: "anthropic/claude-3-haiku",
                TaskTier.CODING: "anthropic/claude-sonnet-4-5",
                TaskTier.REASONING: "anthropic/claude-opus-4-5",
            },
            Provider.GEMINI: {
                TaskTier.FAST: "google/gemini-3-flash-preview",
                TaskTier.CODING: "google/gemini-3-flash-preview",
                TaskTier.REASONING: "google/gemini-3-pro-preview",
            },
        }
        return mapping.get(provider, {}).get(tier, "meta-llama/llama-3.3-70b")
```

**Benefits**:
- Automatic fallback if OpenAI/Gemini/Anthropic quota hit
- Access to DeepSeek R1 (excellent code model)
- Free tier models (Llama 3.3, Qwen) for testing
- Single API for all providers

---

## 3. Mixture of Experts (MoE) Architecture

### Concept: Multiple Specialized Agents

Instead of one monolithic model, use **specialized experts** for different ROM hacking tasks:

```
User Intent
    ↓
Task Classifier (decides which expert to use)
    ↓
┌────────────────┬────────────────┬────────────────┐
│  ASM Expert    │  YAZE Expert   │  Debug Expert  │
│  (fine-tuned)  │  (fine-tuned)  │  (fine-tuned)  │
│                │                │                │
│  65816 asm     │  C++ tools     │  Error fixing  │
│  optimization  │  ROM patching  │  diagnostics   │
└────────────────┴────────────────┴────────────────┘
    ↓
Synthesizer (combines outputs)
    ↓
Final Result
```

### Implementation

**Expert 1: ASM Code Expert**
```toml
[experts.asm_expert]
model = "qwen2.5-coder:14b"
lora_adapter = "~/.context/models/alttp_asm_agent/lora_adapters"
specialization = "65816 assembly, ALTTP routines, memory maps"
routing_keywords = ["asm", "assembly", "routine", "bank", "memory", "optimization"]
confidence_threshold = 0.75
```

**Expert 2: YAZE Tool Expert**
```toml
[experts.yaze_expert]
model = "qwen2.5-coder:14b"
lora_adapter = "~/.context/models/yaze_tool_agent/lora_adapters"
specialization = "YAZE C++ API, ROM manipulation, tool calling"
routing_keywords = ["yaze", "rom", "graphics", "sprite", "map", "tool"]
confidence_threshold = 0.70
```

**Expert 3: Debug Expert**
```toml
[experts.debug_expert]
model = "qwen2.5-coder:14b"
lora_adapter = "~/.context/models/debug_agent/lora_adapters"
specialization = "Error diagnostics, debugging, failure analysis"
routing_keywords = ["error", "bug", "crash", "fix", "debug", "problem"]
confidence_threshold = 0.80
```

**Task Classifier (Routing Agent)**
```python
# src/agents/moe/classifier.py
class TaskClassifier:
    """Routes user intent to appropriate expert."""

    async def classify(self, user_intent: str) -> list[tuple[str, float]]:
        """Classify which expert(s) should handle this task.

        Returns:
            List of (expert_name, confidence) tuples sorted by confidence.
        """
        # Use lightweight model for classification
        prompt = f"""
        Classify this ROM hacking task into categories:
        - asm: Assembly code generation/modification
        - yaze: YAZE tool usage, ROM patching
        - debug: Error fixing, diagnostics
        - multi: Requires multiple experts

        Task: {user_intent}

        Output JSON: {{"experts": ["asm", "yaze"], "confidence": [0.9, 0.6]}}
        """

        response = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.FAST,
            max_tokens=100,
        )

        # Parse and return expert assignments
        return self._parse_classification(response.content)
```

**MoE Orchestrator**
```python
# src/agents/moe/orchestrator.py
class MoEOrchestrator:
    """Mixture of Experts orchestrator for ROM hacking."""

    def __init__(self):
        self.classifier = TaskClassifier()
        self.experts = {
            "asm": AsmExpert(),
            "yaze": YazeExpert(),
            "debug": DebugExpert(),
        }

    async def execute(self, user_intent: str) -> str:
        """Execute task using appropriate expert(s)."""
        # Step 1: Classify task
        classifications = await self.classifier.classify(user_intent)

        # Step 2: Route to expert(s)
        if len(classifications) == 1:
            # Single expert
            expert_name, confidence = classifications[0]
            expert = self.experts[expert_name]
            result = await expert.generate(user_intent)
            return result

        else:
            # Multi-expert collaboration
            results = {}
            for expert_name, confidence in classifications:
                if confidence > 0.6:  # Threshold
                    expert = self.experts[expert_name]
                    results[expert_name] = await expert.generate(user_intent)

            # Step 3: Synthesize results
            return await self._synthesize(user_intent, results)

    async def _synthesize(
        self, user_intent: str, results: dict[str, str]
    ) -> str:
        """Combine outputs from multiple experts."""
        prompt = f"""
        Synthesize these expert outputs into a cohesive solution:

        User Intent: {user_intent}

        Expert Outputs:
        {json.dumps(results, indent=2)}

        Create a unified solution that integrates all expert insights.
        """

        response = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.REASONING,
            max_tokens=2000,
        )

        return response.content
```

**Benefits**:
- **Specialization**: Each expert focuses on one domain
- **Parallelization**: Multiple experts can work simultaneously
- **Modularity**: Easy to add new experts (Oracle Expert, Gigaleak Expert)
- **Quality**: Fine-tuned experts outperform generalist models

**Training Strategy for MoE**:
1. Train ASM Expert on ALTTP ASM dataset (24K samples)
2. Train YAZE Expert on YAZE tools dataset (7K samples)
3. Train Debug Expert on error diagnostics (1.5K samples)
4. Train Classifier on task routing examples (500 samples)

Total: **33K samples** (already in pipeline!)

---

## 4. Model Name Ideas

### Naming Convention
`{project}-{domain}-{version}`

### Recommended Names

**ALTTP ASM Agent**:
- `hyrule-asm-v1` (Hyrule = ALTTP kingdom)
- `triforce-coder-v1` (Triforce = 3 strengths: ASM, ROM, Debug)
- `master-sword-14b` (Master Sword = legendary item)
- `lightworld-asm-v1` (Light World vs Dark World)

**YAZE Tool Agent**:
- `yaze-sage-v1` (Sage = wise tool user)
- `rom-artificer-v1` (Artificer = craftsperson)
- `tool-herald-v1` (Herald = messenger/announcer)
- `patch-smith-v1` (Smith = crafts patches)

**Debug Agent**:
- `debug-oracle-v1` (Oracle = knows all, sees errors)
- `error-sage-v1` (Sage of Error Handling)
- `bug-hunter-v1` (Bug Hunter = fixes issues)

**MoE System**:
- `triforce-moe-v1` (3 experts like 3 Triforce pieces)
- `hyrule-council-v1` (Council of experts)
- `sage-collective-v1` (7 Sages in OoT)

### Full Model Registry

```toml
# ~/.context/models/registry.toml
[models.hyrule-asm-v1]
base = "qwen2.5-coder:14b"
adapter_path = "~/.context/models/alttp_asm_agent/lora_adapters"
training_date = "2025-12-21"
dataset = "alttp_asm_24k"
specialization = "65816 assembly, ALTTP routines"
test_perplexity = 2.8
test_accuracy = 0.42
version = "1.0.0"

[models.yaze-sage-v1]
base = "qwen2.5-coder:14b"
adapter_path = "~/.context/models/yaze_tool_agent/lora_adapters"
training_date = "2025-12-21"
dataset = "yaze_tools_7k"
specialization = "YAZE C++ tools, ROM manipulation"
test_perplexity = 3.2
test_accuracy = 0.38
version = "1.0.0"

[models.triforce-moe-v1]
type = "mixture_of_experts"
experts = ["hyrule-asm-v1", "yaze-sage-v1", "debug-oracle-v1"]
classifier = "task-router-v1"
version = "1.0.0"
```

**Recommendation**: Use **`hyrule-asm-v1`** and **`yaze-sage-v1`** for initial release.

---

## 5. Training Resilience Strategies

### Problem: What if training fails?

**Common Failure Modes**:
1. VRAM overflow (OOM errors)
2. Dataset corruption
3. Loss divergence (NaN loss)
4. Checkpoint corruption
5. Hardware failure

### Mitigation Strategies

#### **Strategy 1: Checkpoint Every Epoch**
```python
# src/agents/training/trainer.py
class ResilientTrainer:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 100  # Save every 100 steps

    async def train(self):
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.dataloader):
                # Train step
                loss = await self.train_step(batch)

                # Checkpoint every N steps
                if step % self.checkpoint_interval == 0:
                    await self.save_checkpoint(epoch, step, loss)

            # Checkpoint at end of epoch
            await self.save_checkpoint(epoch, "final", loss)

    async def save_checkpoint(self, epoch: int, step: int, loss: float):
        """Save checkpoint with metadata."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "timestamp": time.time(),
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    async def resume_from_checkpoint(self, checkpoint_path: Path):
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.start_step = checkpoint["step"]

        logger.info(f"Resumed from epoch {self.start_epoch}, step {self.start_step}")
```

#### **Strategy 2: Automatic VRAM Detection**
```python
# src/agents/training/auto_batch.py
class AutoBatchSizer:
    """Automatically find optimal batch size for available VRAM."""

    async def find_optimal_batch_size(self, model: nn.Module) -> int:
        """Binary search for max batch size that fits in VRAM."""
        min_batch = 1
        max_batch = 32
        optimal = 1

        while min_batch <= max_batch:
            mid = (min_batch + max_batch) // 2

            try:
                # Test if batch size fits
                success = await self._test_batch_size(model, mid)

                if success:
                    optimal = mid
                    min_batch = mid + 1
                else:
                    max_batch = mid - 1

            except torch.cuda.OutOfMemoryError:
                max_batch = mid - 1

        logger.info(f"Optimal batch size: {optimal}")
        return optimal

    async def _test_batch_size(self, model: nn.Module, batch_size: int) -> bool:
        """Test if batch size fits in VRAM."""
        try:
            # Create dummy batch
            dummy_input = torch.randn(batch_size, 2048).cuda()

            # Forward + backward pass
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()

            # Clear cache
            del dummy_input, output, loss
            torch.cuda.empty_cache()

            return True

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
```

#### **Strategy 3: Loss Monitoring & Early Stopping**
```python
# src/agents/training/monitor.py
class LossMonitor:
    """Monitor training loss for anomalies."""

    def __init__(self, patience: int = 5):
        self.losses: list[float] = []
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def check_loss(self, loss: float) -> tuple[bool, str]:
        """Check if loss is healthy.

        Returns:
            (is_healthy, message)
        """
        self.losses.append(loss)

        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            return (False, f"Loss is {loss}, training diverged")

        # Check for explosion (loss > 10x median)
        if len(self.losses) > 10:
            median_loss = statistics.median(self.losses[-10:])
            if loss > median_loss * 10:
                return (False, f"Loss exploded: {loss} >> {median_loss}")

        # Early stopping check
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            return (False, f"No improvement for {self.patience} epochs, early stopping")

        return (True, "OK")
```

#### **Strategy 4: Dataset Validation Pre-Training**
```python
# src/agents/training/validate.py
class DatasetValidator:
    """Validate dataset before training starts."""

    async def validate(self, dataset_path: Path) -> bool:
        """Check dataset integrity."""
        logger.info(f"Validating dataset: {dataset_path}")

        # Check file exists
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return False

        # Load dataset
        with open(dataset_path) as f:
            samples = [json.loads(line) for line in f]

        # Validate structure
        for i, sample in enumerate(samples):
            if not self._validate_sample(sample):
                logger.error(f"Invalid sample at line {i}: {sample}")
                return False

        # Check for duplicates
        unique_hashes = set()
        duplicates = 0
        for sample in samples:
            hash_key = hashlib.md5(
                json.dumps(sample, sort_keys=True).encode()
            ).hexdigest()
            if hash_key in unique_hashes:
                duplicates += 1
            unique_hashes.add(hash_key)

        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate samples")

        logger.info(f"✓ Dataset validated: {len(samples)} samples, {duplicates} duplicates")
        return True

    def _validate_sample(self, sample: dict) -> bool:
        """Check sample has required fields."""
        required = ["instruction", "input", "output"]
        return all(field in sample for field in required)
```

#### **Strategy 5: Multi-Stage Training**
```toml
# Train in stages with validation between
[training.stage1]
name = "warmup"
samples = 1000
epochs = 1
learning_rate = 1e-5
validation_after = true

[training.stage2]
name = "main"
samples = "all"
epochs = 2
learning_rate = 5e-5
validation_after = true

[training.stage3]
name = "cooldown"
samples = 1000
epochs = 1
learning_rate = 1e-6
validation_after = true
```

**Benefits**:
- If stage 1 fails, no time wasted
- Early validation catches issues
- Gradual learning rate warmup

---

## 6. Additional Preparation Checklist

### Pre-Training
- [ ] Validate all 34.5K samples for correctness
- [ ] Check for duplicate samples (deduplication)
- [ ] Verify VRAM available on medical-mechanica
- [ ] Test checkpoint save/load on small dataset (100 samples)
- [ ] Set up Weights & Biases or TensorBoard logging
- [ ] Create validation split (10% of data)

### During Training
- [ ] Monitor loss curves in real-time
- [ ] Check GPU utilization (should be >90%)
- [ ] Watch for OOM errors
- [ ] Validate checkpoints every epoch
- [ ] Test inference with trained model at epoch 1

### Post-Training
- [ ] Evaluate on held-out test set
- [ ] Calculate perplexity and accuracy metrics
- [ ] Compare fine-tuned vs base model
- [ ] Test on real ROM hacking tasks
- [ ] Deploy to production if metrics pass threshold

### Backup Strategy
```bash
# Backup checkpoints to cloud storage
rclone sync ~/.context/models/alttp_asm_agent/ \
    gdrive:hafs/models/alttp_asm_agent/

# Backup datasets
rclone sync ~/.context/training/datasets/ \
    gdrive:hafs/datasets/
```

---

## 7. Quick Start Commands

### Test OpenRouter
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python -m hafs.scripts.test_openrouter
```

### Train with Auto-Recovery
```bash
# Start training with automatic checkpointing
python -m agents.training.scripts.train_resilient \
    --dataset alttp_asm_24k \
    --model qwen2.5-coder:14b \
    --output hyrule-asm-v1 \
    --checkpoint-interval 100 \
    --resume-from-latest  # Auto-resume if interrupted
```

### Deploy MoE System
```bash
# Test MoE orchestrator
python -m agents.moe.test_orchestrator \
    --experts asm,yaze,debug \
    --intent "Add a new item to ALTTP that uses custom graphics from YAZE"
```

### Monitor Training
```bash
# Real-time training dashboard
hafs training dashboard --model hyrule-asm-v1
```

---

## Summary

**Hardware**: ✓ medical-mechanica can train Qwen2.5-Coder:14B with 4-bit LoRA (12-16 hours)

**OpenRouter**: Recommended for quota management and access to DeepSeek R1

**MoE**: Feasible with current infrastructure, provides better specialization

**Model Names**: `hyrule-asm-v1` and `yaze-sage-v1` for initial release

**Resilience**: Checkpoint every 100 steps, auto-resume, loss monitoring, dataset validation

**Next Steps**:
1. Implement OpenRouter backend (2 hours)
2. Set up checkpoint system (1 hour)
3. Validate 34.5K dataset (30 min)
4. Start training on medical-mechanica (12-16 hours)
5. Evaluate and deploy MoE if needed (4 hours)
