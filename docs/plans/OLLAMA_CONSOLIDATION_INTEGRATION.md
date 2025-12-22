# Ollama Integration for Filesystem Consolidation Analyzer

**Created:** 2025-12-21
**Status:** Design Phase
**Target:** Add local AI recommendations to consolidation_analyzer.py

## Overview

Enhance the filesystem consolidation analyzer with local AI-powered recommendations using the existing Ollama infrastructure. This enables intelligent, context-aware file organization suggestions without requiring GPU resources or API costs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Filesystem Scan Pipeline                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  filesystem_explorer.py (Python)                                │
│  • Scans C:/D:/E: drives                                        │
│  • MD5 hashing for duplicates                                   │
│  • Creates JSON inventory                                       │
│  Output: filesystem_inventory_20251221.json                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  consolidation_analyzer.py (Python + Ollama)                    │
│                                                                 │
│  Phase 1: Rule-Based Analysis                                  │
│  • Duplicate detection                                          │
│  • Extension categorization                                     │
│  • Organization patterns                                        │
│                                                                 │
│  Phase 2: AI Enhancement (NEW)                                 │
│  • Summarize inventory for LLM                                  │
│  • Call local Ollama API (CPU only)                             │
│  • Generate context-aware recommendations                       │
│                                                                 │
│  Output: consolidation_report.md + ai_recommendations.md        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Reports → Mac via Tailscale                                    │
│  ~/Mounts/mm-d/.context/scratchpad/consolidation_analyzer/      │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### 1. Add Ollama Client Helper

```python
# src/agents/background/ollama_client.py

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for local Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API endpoint
        """
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running.

        Returns:
            True if Ollama API is accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def generate(
        self,
        prompt: str,
        model: str = "qwen2.5:3b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """Generate text using Ollama.

        Args:
            prompt: Input prompt
            model: Model name (default: qwen2.5:3b for speed)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None on error
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,  # 2 minute timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return None
```

### 2. Enhance ConsolidationAnalyzerAgent

Add new method to `src/agents/background/consolidation_analyzer.py`:

```python
from agents.background.ollama_client import OllamaClient

class ConsolidationAnalyzerAgent(BackgroundAgent):
    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        super().__init__(config_path, verbose)
        self.inventory_dir = Path(...)
        self.consolidation_rules = self.config.tasks.get("consolidation_rules", {})

        # NEW: Initialize Ollama client
        self.use_ai_recommendations = self.config.tasks.get("use_ai_recommendations", True)
        self.ai_model = self.config.tasks.get("ai_model", "qwen2.5:3b")
        self.ollama_client = OllamaClient() if self.use_ai_recommendations else None

    def run(self) -> dict[str, Any]:
        """Execute consolidation analysis."""
        results = {...}

        # ... existing analysis ...

        # NEW: Generate AI recommendations
        if self.use_ai_recommendations and self.ollama_client:
            if self.ollama_client.is_available():
                logger.info("Generating AI-powered recommendations...")
                ai_recommendations = self._generate_ai_recommendations(inventory, results)
                results["ai_recommendations"] = ai_recommendations

                # Save AI recommendations separately
                self._save_output(
                    {"timestamp": results["analysis_timestamp"], "recommendations": ai_recommendations},
                    "ai_recommendations"
                )
            else:
                logger.warning("Ollama not available, skipping AI recommendations")

        return results

    def _generate_ai_recommendations(
        self,
        inventory: dict[str, Any],
        analysis: dict[str, Any]
    ) -> str:
        """Generate AI-powered recommendations using local Ollama.

        Args:
            inventory: Raw filesystem inventory
            analysis: Rule-based analysis results

        Returns:
            AI-generated recommendations as markdown
        """
        # Create summary for LLM (keep token count low)
        summary = self._create_inventory_summary(inventory, analysis)

        # Build prompt
        prompt = f"""You are a filesystem organization expert. Based on the following filesystem analysis, provide intelligent recommendations for organizing and consolidating files.

## Filesystem Summary

{summary}

## Your Task

Provide 5-10 specific, actionable recommendations for:
1. File organization strategies (e.g., creating project-specific directories)
2. Duplicate cleanup priorities (what to review first)
3. Archival candidates (old files that could be compressed/moved)
4. Potential waste reduction (temp files, caches, etc.)
5. Long-term organization structure

Format your response as markdown with clear headings and bullet points. Be specific and reference actual file types/sizes from the summary above.

Focus on HIGH-IMPACT actions that will save the most space or improve organization the most.
"""

        # Call Ollama
        response = self.ollama_client.generate(
            prompt=prompt,
            model=self.ai_model,
            temperature=0.7,
            max_tokens=2048,
        )

        if not response:
            return "AI recommendations unavailable (Ollama error)"

        return response

    def _create_inventory_summary(
        self,
        inventory: dict[str, Any],
        analysis: dict[str, Any]
    ) -> str:
        """Create concise summary for LLM.

        Args:
            inventory: Full inventory
            analysis: Analysis results

        Returns:
            Markdown summary
        """
        lines = []

        # Overall stats
        lines.append("### Overall Statistics")
        lines.append(f"- **Total Files:** {inventory['total_files']:,}")
        lines.append(f"- **Total Size:** {inventory['total_size_gb']:.1f} GB")
        lines.append(f"- **Directories Scanned:** {len(inventory['directory_stats'])}")
        lines.append("")

        # Duplicate analysis
        if inventory.get("duplicate_candidates"):
            lines.append("### Duplicates")
            total_waste = sum(d["waste_mb"] for d in inventory["duplicate_candidates"]) / 1024
            lines.append(f"- **Duplicate Groups:** {len(inventory['duplicate_candidates'])}")
            lines.append(f"- **Potential Savings:** {total_waste:.2f} GB")

            # Top 5 duplicate groups
            top_dupes = sorted(
                inventory["duplicate_candidates"],
                key=lambda x: x["waste_mb"],
                reverse=True
            )[:5]

            lines.append("- **Top Duplicates:**")
            for dup in top_dupes:
                ext = Path(dup["files"][0]).suffix or "no_ext"
                lines.append(f"  - {ext}: {dup['count']} copies, {dup['waste_mb']:.1f} MB waste")
            lines.append("")

        # Extension summary (top 10 by size)
        if inventory.get("extension_summary"):
            lines.append("### File Types (Top 10 by Size)")
            ext_by_size = sorted(
                inventory["extension_summary"].items(),
                key=lambda x: x[1]["size_gb"],
                reverse=True
            )[:10]

            for ext, data in ext_by_size:
                lines.append(f"- **{ext}**: {data['count']:,} files, {data['size_gb']:.2f} GB")
            lines.append("")

        # Directory breakdown
        if inventory.get("directory_stats"):
            lines.append("### Directory Overview")
            for dir_stat in inventory["directory_stats"][:3]:  # Top 3 scan paths
                lines.append(f"- **{dir_stat['path']}**")
                lines.append(f"  - Files: {dir_stat['total_files']:,}")
                lines.append(f"  - Size: {dir_stat['total_size_gb']:.2f} GB")
            lines.append("")

        # Rule-based recommendations summary
        if analysis.get("priority_actions"):
            lines.append("### Current Recommendations (Rule-Based)")
            for i, action in enumerate(analysis["priority_actions"][:5], 1):
                lines.append(f"{i}. {action['description']} ({action.get('savings_gb', 0):.2f} GB)")
            lines.append("")

        return "\n".join(lines)
```

### 3. Update Configuration

Add to `config/windows_filesystem_agents.toml`:

```toml
[agents.consolidationanalyzer.tasks]
inventory_dir = "D:/.context/scratchpad/filesystem_explorer"
output_dir = "D:/.context/scratchpad/consolidation_analyzer"
report_dir = "D:/.context/logs/consolidation_analyzer"

# NEW: AI recommendations
use_ai_recommendations = true
ai_model = "qwen2.5:3b"  # Fast 3B model for CPU inference
ollama_url = "http://localhost:11434"

[agents.consolidation_analyzer.tasks.consolidation_rules]
min_duplicate_savings_gb = 0.1
min_category_size_gb = 0.5
organize_code = "D:/projects/code"
organize_data = "D:/projects/data"
organize_models = "D:/.context/training/models"
organize_datasets = "D:/.context/training/datasets"
```

### 4. Enhanced Report Generation

Update `_generate_report()` to include AI recommendations:

```python
def _generate_report(self, results: dict[str, Any]) -> str:
    """Generate markdown report with AI recommendations."""
    lines = [
        "# Filesystem Consolidation Report",
        "",
        f"**Generated:** {results['analysis_timestamp']}",
        f"**Potential Savings:** {results['potential_savings_gb']:.2f} GB",
        "",
    ]

    # AI Recommendations section (NEW)
    if results.get("ai_recommendations"):
        lines.extend([
            "## 🤖 AI-Powered Recommendations",
            "",
            results["ai_recommendations"],
            "",
            "---",
            "",
        ])

    # ... existing priority actions and organization suggestions ...

    return "\n".join(lines)
```

## Resource Usage

### CPU-Only Operation
- **Model:** qwen2.5:3b (~3.5GB RAM)
- **Inference Time:** ~30-60 seconds per report
- **GPU Impact:** ZERO (runs on CPU only)
- **Training Impact:** Minimal (runs after filesystem scan completes)

### Alternative Models
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| qwen2.5:3b | 3.5GB | Fast | Good | Default (fast recommendations) |
| llama3.2:3b | 3.5GB | Fast | Good | Alternative lightweight |
| qwen2.5:7b | 7.5GB | Medium | Better | More detailed analysis |
| llama3.1:8b | 8GB | Medium | Better | High-quality recommendations |

## Example Output

### Input Summary
```markdown
### Overall Statistics
- Total Files: 43,799
- Total Size: 3,421.5 GB
- Directories Scanned: 10,000

### Duplicates
- Duplicate Groups: 234
- Potential Savings: 12.5 GB
- Top Duplicates:
  - .png: 45 copies, 2.3 MB waste
  - .json: 89 copies, 1.8 MB waste

### File Types (Top 10 by Size)
- .safetensors: 15 files, 125.3 GB
- .bin: 234 files, 89.7 GB
- .jsonl: 456 files, 45.2 GB
```

### AI Recommendations Output
```markdown
## 🤖 AI-Powered Recommendations

Based on the filesystem analysis, here are my recommendations:

### 1. Model Organization Strategy

Your largest files are ML models (.safetensors, .bin) totaling 215 GB. Recommendations:

- **Create dedicated model repository**: `D:/projects/ml-models/`
- **Organize by framework**: pytorch/, tensorflow/, onnx/
- **Archive old checkpoints**: Compress models older than 90 days
- **Potential savings**: ~50 GB from deduplication and compression

### 2. Training Dataset Consolidation

You have 456 .jsonl files (45.2 GB) scattered across multiple locations:

- **Consolidate to**: `D:/.context/training/datasets/`
- **Separate by domain**: asm/, oracle/, yaze/
- **Remove test samples**: Keep only production-quality datasets
- **Potential savings**: ~15 GB from removing test/debug data

### 3. Duplicate Cleanup Priority

Focus on these high-impact duplicate cleanups:

1. **PNG images** (45 duplicates, 2.3 MB): Likely icons/assets - keep one copy in shared assets folder
2. **JSON configs** (89 duplicates, 1.8 MB): Review config files, many likely from git repos
3. **Python caches** (__pycache__/): Safe to delete all, regenerate on demand

### 4. Temporary File Cleanup

Identified several categories of temporary files safe to remove:

- **Build artifacts**: /build/, /dist/ directories
- **Python caches**: __pycache__/, *.pyc
- **Node modules**: node_modules/ (can reinstall from package.json)
- **Estimated savings**: ~8 GB

### 5. Long-Term Organization Structure

Recommend this directory hierarchy for D: drive:

```
D:/
├── projects/
│   ├── code/           # Git repos, source code
│   ├── data/           # Datasets, raw data
│   ├── ml-models/      # Trained models
│   └── archives/       # Compressed backups
├── .context/
│   ├── training/       # Active training data
│   ├── logs/           # System logs
│   └── scratchpad/     # Temporary workspace
└── personal/           # Documents, media
```

This structure separates active development from archived content and makes backups more efficient.
```

## Testing Plan

### Phase 1: Local Testing (Mac)
1. Mock Ollama responses for testing
2. Verify prompt engineering
3. Test summary generation logic

### Phase 2: Windows Integration
1. Ensure Ollama is installed and running
2. Test with real filesystem inventory
3. Verify CPU-only operation (no GPU interference)
4. Monitor resource usage during training

### Phase 3: Validation
1. Compare AI recommendations vs rule-based
2. Verify actionability of suggestions
3. Measure time overhead
4. Gather user feedback

## Benefits

### Intelligent Context
- **Beyond Rules**: AI understands semantic relationships between files
- **Domain Awareness**: Can suggest project-specific organization
- **Prioritization**: Ranks actions by real-world impact

### Cost Effective
- **No API Costs**: Runs entirely local
- **No GPU**: Uses CPU only, won't affect training
- **Fast**: 30-60 seconds per report

### User Experience
- **Natural Language**: Recommendations in plain English
- **Actionable**: Specific file paths and commands
- **Educational**: Explains reasoning behind suggestions

## Future Enhancements

1. **Interactive Mode**: User can ask follow-up questions about recommendations
2. **Auto-Execution**: Optionally apply safe recommendations automatically (with confirmation)
3. **Learning**: Track which recommendations users follow to improve prompts
4. **Multi-Model Support**: Use different models for different analysis types
5. **Streaming**: Show recommendations as they're generated for long analyses

## Implementation Timeline

1. **Create OllamaClient** (~30 min)
2. **Add AI recommendation method** (~1 hour)
3. **Update configuration** (~15 min)
4. **Test with mock data** (~30 min)
5. **Deploy to Windows** (~15 min)
6. **Validate with real scan** (~1 hour)

**Total:** ~3.5 hours

## Success Metrics

- ✅ AI recommendations generated in <2 minutes
- ✅ Zero GPU interference with training
- ✅ Recommendations are actionable (include specific paths/commands)
- ✅ User finds ≥3 useful suggestions per report
- ✅ No crashes or errors when Ollama unavailable (graceful fallback)
