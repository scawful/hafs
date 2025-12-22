# Cognitive Protocol Configuration Guide

**Last Updated**: 2025-12-22
**Config Version**: 1.0
**For**: hAFS v0.5.0+

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Structure](#configuration-structure)
3. [Personality Profiles](#personality-profiles)
4. [Tuning by Section](#tuning-by-section)
5. [CLI Commands](#cli-commands)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Default Configuration

The system works out-of-the-box with sensible defaults:

```python
from core.config.loader import get_config

# Load default configuration
config = get_config()
```

### Using Personality Profiles

For quick adjustments, use pre-built personality profiles:

```python
# Conservative agent (asks for help early, detects spinning sooner)
config = get_config(personality="cautious")

# Risk-tolerant agent (rarely asks for help, tolerates more iteration)
config = get_config(personality="aggressive")

# Exploration-focused agent (tracks more items, slower knowledge decay)
config = get_config(personality="researcher")

# Implementation-focused agent (tolerates prototyping, lenient on tests)
config = get_config(personality="builder")

# Quality-focused agent (strict about testing, low error tolerance)
config = get_config(personality="critic")
```

### CLI Usage

```bash
# View current configuration
hafs config show

# View specific section
hafs config show --section metacognition

# View with personality
hafs config show --personality cautious

# List available personalities
hafs config list-personalities

# Compare personalities
hafs config compare cautious aggressive

# Validate custom config
hafs config validate my_config.toml
```

---

## Configuration Structure

### File Locations

```
config/
├── cognitive_protocol.toml        # Base configuration (DO NOT EDIT directly)
└── agent_personalities.toml       # Personality profiles (SAFE to edit)
```

See full guide at: `docs/guides/COGNITIVE_CONFIG_GUIDE.md`

For complete documentation of all 60+ parameters, tuning guides, personality profiles, troubleshooting, and migration strategies, please refer to the full configuration guide.

**Quick Links**:
- Architecture: `docs/architecture/COGNITIVE_PROTOCOL.md`
- Training Models: `docs/training/MODEL_TRAINING_PLAN.md`
- MoE System: `docs/architecture/MOE_SYSTEM.md`
