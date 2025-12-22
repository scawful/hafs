# hAFS Documentation Index

**Last Updated**: 2025-12-22
**Version**: 0.5.0

---

## Quick Start

**New to hAFS?** Start here:

1. [Usage Guide](guides/USAGE.md) - Basic usage and commands
2. [Configuration Guide](guides/CONFIGURATION.md) - System configuration
3. [Agents Quickstart](guides/AGENTS_QUICKSTART.md) - Agent setup and basics
4. [Agentic System Guide](guides/AGENTIC_SYSTEM_GUIDE.md) - Multi-agent architecture overview

---

## Recent Additions (2025-12-22)

**NEW** - Phase 5 Cognitive Protocol Documentation:

- **[Cognitive Protocol Architecture](architecture/COGNITIVE_PROTOCOL.md)** - Complete architecture reference for the cognitive protocol system (Read-Process-Update-Reflect loop, IOManager, schema validation)
- **[Cognitive Config Guide](guides/COGNITIVE_CONFIG_GUIDE.md)** - Configuration guide for cognitive protocol tuning and personality profiles
- **[Oracle-Farore-Secrets Plan](training/ORACLE_FARORE_SECRETS_PLAN.md)** - North star unified model plan (32B Qwen2.5-Coder, data difficulty assessment, HITL requirements)
- **[AFS Live Source Plan](architecture/AFS_LIVE_SOURCE_PLAN.md)** - Live code source integration plan
- **[Training Snapshot Schema](architecture/TRAINING_SNAPSHOT_SCHEMA.md)** - Schema for training snapshots

---

## Architecture

Core system design and technical architecture:

- **[ARCHITECTURE.md](architecture/ARCHITECTURE.md)** - Overall system architecture
- **[COGNITIVE_PROTOCOL.md](architecture/COGNITIVE_PROTOCOL.md)** - **NEW** Cognitive protocol architecture (file system, data flow, config system, I/O optimization)
- **[MOE_SYSTEM.md](architecture/MOE_SYSTEM.md)** - Mixture of Experts routing and agent coordination (2025-12-21)
- **[AFS_CONTRACT.md](architecture/AFS_CONTRACT.md)** - Agent File System contract and guarantees
- **[AFS_LIVE_SOURCE_PLAN.md](architecture/AFS_LIVE_SOURCE_PLAN.md)** - **NEW** Plan for live code source integration
- **[TRAINING_SNAPSHOT_SCHEMA.md](architecture/TRAINING_SNAPSHOT_SCHEMA.md)** - **NEW** Training snapshot schema documentation
- **[SPRINT5_ARCHITECTURE_DIAGRAM.md](architecture/SPRINT5_ARCHITECTURE_DIAGRAM.md)** - Sprint 5 architecture diagram
- **[native-similarity.md](architecture/native-similarity.md)** - Native similarity search implementation

---

## Guides

### Agent System
- **[AGENT_REFERENCE.md](guides/AGENT_REFERENCE.md)** - Complete agent reference
- **[AGENTIC_SYSTEM_GUIDE.md](guides/AGENTIC_SYSTEM_GUIDE.md)** - Comprehensive multi-agent system guide
- **[AGENTS_QUICKSTART.md](guides/AGENTS_QUICKSTART.md)** - Quick start for agents
- **[agents.md](guides/agents.md)** - Agent basics
- **[AGENT_MESSAGE_DELIVERY.md](guides/AGENT_MESSAGE_DELIVERY.md)** - Inter-agent messaging

### Configuration
- **[CONFIGURATION.md](guides/CONFIGURATION.md)** - Main system configuration guide
- **[COGNITIVE_CONFIG_GUIDE.md](guides/COGNITIVE_CONFIG_GUIDE.md)** - **NEW** Cognitive protocol configuration (personalities, tuning)
- **[SWITCHING_MODELS.md](guides/SWITCHING_MODELS.md)** - How to switch between models

### Development
- **[USAGE.md](guides/USAGE.md)** - Basic usage guide
- **[SKILLS.md](guides/SKILLS.md)** - Skills system documentation
- **[TRAINING_DEVELOPMENT_WORKFLOW.md](guides/TRAINING_DEVELOPMENT_WORKFLOW.md)** - Training development workflow
- **[WEBSITE_MONITORING_AGENTS.md](guides/WEBSITE_MONITORING_AGENTS.md)** - Website monitoring agent setup

### LSP Integration
- **[LSP_SETUP.md](guides/LSP_SETUP.md)** - Language Server Protocol setup
- **[HAFS_LSP_CONTROL.md](guides/HAFS_LSP_CONTROL.md)** - LSP control mechanisms
- **[HAFS_LSP_QUICKSTART.md](guides/HAFS_LSP_QUICKSTART.md)** - Quick start for LSP

### Analysis
- **[REJECTED_SAMPLES_ANALYSIS.md](guides/REJECTED_SAMPLES_ANALYSIS.md)** - Training sample rejection analysis

---

## Training

Machine learning model training and evaluation:

### Core Training Docs (MOST RECENT)
- **[ORACLE_FARORE_SECRETS_PLAN.md](training/ORACLE_FARORE_SECRETS_PLAN.md)** - **NEW** North star unified model (32B, data difficulty, HITL, 20-week timeline)
- **[MODEL_TRAINING_PLAN.md](training/MODEL_TRAINING_PLAN.md)** - Master training plan (2025-12-22)
- **[GPU_LIMITATIONS.md](training/GPU_LIMITATIONS.md)** - RTX 5060 Ti support, workarounds (2025-12-22)
- **[AUTONOMOUS_TRAINING_README.md](training/AUTONOMOUS_TRAINING_README.md)** - Autonomous training overview

### Training Infrastructure
- **[HYBRID_TRAINING_INFRASTRUCTURE.md](training/HYBRID_TRAINING_INFRASTRUCTURE.md)** - Mac + Windows hybrid training setup
- **[AUTONOMOUS_TRAINING.md](training/AUTONOMOUS_TRAINING.md)** - Autonomous training system
- **[CONFIG_BASED_TRAINING.md](training/CONFIG_BASED_TRAINING.md)** - Configuration-driven training
- **[TRAINING_PIPELINE_OVERVIEW.md](training/TRAINING_PIPELINE_OVERVIEW.md)** - Complete pipeline overview
- **[TRAINING_PREPARATION.md](training/TRAINING_PREPARATION.md)** - Training preparation steps
- **[training_pipeline_improvements.md](training/training_pipeline_improvements.md)** - Pipeline improvements

### Training Methods
- **[LORA_TRAINING.md](training/LORA_TRAINING.md)** - LoRA fine-tuning guide
- **[RLHF_DPO.md](training/RLHF_DPO.md)** - RLHF and DPO training
- **[GPU_ACCELERATION.md](training/GPU_ACCELERATION.md)** - GPU acceleration setup

### Quality & Evaluation
- **[EVALUATION.md](training/EVALUATION.md)** - Model evaluation framework
- **[AB_TESTING_GUIDE.md](training/AB_TESTING_GUIDE.md)** - A/B testing methodology
- **[QUALITY_IMPROVEMENT_GUIDE.md](training/QUALITY_IMPROVEMENT_GUIDE.md)** - Quality improvement strategies
- **[QUALITY_IMPROVEMENT_PATH.md](training/QUALITY_IMPROVEMENT_PATH.md)** - Quality improvement roadmap

### Platform-Specific
- **[WINDOWS_MODELS_SETUP.md](training/WINDOWS_MODELS_SETUP.md)** - Windows model setup
- **[CAMPAIGN_LAUNCH_SUMMARY.md](training/CAMPAIGN_LAUNCH_SUMMARY.md)** - Training campaign summary

---

## Protocols

Agent File System and cognitive protocols:

- **[PROTOCOL_SPEC.md](protocols/PROTOCOL_SPEC.md)** - AFS protocol specification
- **[COGNITIVE_PROTOCOL.md](protocols/COGNITIVE_PROTOCOL.md)** - Cognitive protocol v0.2 (older)
- **[COGNITIVE_PROTOCOL_v0.3.md](protocols/COGNITIVE_PROTOCOL_v0.3.md)** - Cognitive protocol v0.3 spec
- **[COGNITIVE_PROTOCOL_IMPLEMENTATION_REVIEW.md](protocols/COGNITIVE_PROTOCOL_IMPLEMENTATION_REVIEW.md)** - Implementation review
- **[AFS_POLICY_MANAGEMENT.md](protocols/AFS_POLICY_MANAGEMENT.md)** - AFS policy management

**Note**: For latest cognitive protocol architecture, see [architecture/COGNITIVE_PROTOCOL.md](architecture/COGNITIVE_PROTOCOL.md)

---

## Swarm

Multi-agent swarm coordination:

- **[SWARM_README.md](swarm/SWARM_README.md)** - Swarm system overview
- **[SWARM_USAGE.md](swarm/SWARM_USAGE.md)** - Swarm usage guide
- **[SWARM_MISSIONS.md](swarm/SWARM_MISSIONS.md)** - Swarm mission definitions

---

## Plugins

Plugin development and adaptation:

- **[PLUGIN_DEVELOPMENT.md](plugins/PLUGIN_DEVELOPMENT.md)** - Plugin development guide
- **[PLUGIN_ADAPTER_GUIDE.md](plugins/PLUGIN_ADAPTER_GUIDE.md)** - Adapter pattern guide
- **[PLUGIN_ADAPTER_PATTERN.md](plugins/PLUGIN_ADAPTER_PATTERN.md)** - Adapter pattern reference
- **[PLUGIN_AGENT_ADAPTATION_GUIDE.md](plugins/PLUGIN_AGENT_ADAPTATION_GUIDE.md)** - Agent adaptation guide
- **[QUICK_START.md](plugins/QUICK_START.md)** - Plugin quick start
- **[HAFS_SCAWFUL_README.md](plugins/HAFS_SCAWFUL_README.md)** - hafs_scawful plugin README
- **[FILES_TO_MIGRATE.md](plugins/FILES_TO_MIGRATE.md)** - Migration tracking
- **[ZELDA_MODEL_NAMES.md](plugins/ZELDA_MODEL_NAMES.md)** - Zelda-themed model naming

---

## Windows Setup

Windows-specific configuration and deployment:

- **[WINDOWS_SETUP.md](windows/WINDOWS_SETUP.md)** - Windows environment setup
- **[BACKGROUND_AGENTS.md](windows/BACKGROUND_AGENTS.md)** - Background agent setup
- **[BACKGROUND_AGENTS_DEPLOYMENT.md](windows/BACKGROUND_AGENTS_DEPLOYMENT.md)** - Deployment guide
- **[GIT_SYNC_STRATEGY.md](windows/GIT_SYNC_STRATEGY.md)** - Git synchronization strategy
- **[MEDICAL_MECHANICA_SETUP_SUMMARY.md](windows/MEDICAL_MECHANICA_SETUP_SUMMARY.md)** - Medical Mechanica (Windows GPU server) setup

---

## Plans

Future work and active planning documents:

- **[LOCAL_AI_ORCHESTRATION.md](plans/LOCAL_AI_ORCHESTRATION.md)** - Local AI orchestration plan
- **[OLLAMA_CONSOLIDATION_INTEGRATION.md](plans/OLLAMA_CONSOLIDATION_INTEGRATION.md)** - Ollama integration plan
- **[NEXT_STEPS.md](plans/NEXT_STEPS.md)** - Next steps roadmap
- **[CHAT_OVERHAUL_PLAN.md](plans/CHAT_OVERHAUL_PLAN.md)** - Chat interface redesign
- **[CHAT_MODE_RENDERER_PLAN.md](plans/CHAT_MODE_RENDERER_PLAN.md)** - Chat renderer plan
- **[RESEARCH_ALIGNMENT_PLAN.md](plans/RESEARCH_ALIGNMENT_PLAN.md)** - Research alignment strategy
- **[CODE_CONSOLIDATION_PLAN.md](plans/CODE_CONSOLIDATION_PLAN.md)** - Code cleanup plan
- **[SPRINT5_CHAT_MODE_INTEGRATION.md](plans/SPRINT5_CHAT_MODE_INTEGRATION.md)** - Sprint 5 chat integration

---

## Reports

Status reports and analysis:

- **[SPRINT5_SUMMARY.md](reports/SPRINT5_SUMMARY.md)** - Sprint 5 completion summary
- **[CODE_CONSOLIDATION_STATUS.md](reports/CODE_CONSOLIDATION_STATUS.md)** - Code consolidation progress
- **[CODE_QUALITY_REPORT.md](reports/CODE_QUALITY_REPORT.md)** - Code quality analysis
- **[project_history_analysis.md](reports/project_history_analysis.md)** - Project history analysis
- **[ml_visualization.md](reports/ml_visualization.md)** - ML visualization report
- **[ml_visualization_implementation.md](reports/ml_visualization_implementation.md)** - ML viz implementation

---

## Status

Current system status (check for latest updates):

- **[SYSTEM_STATUS.md](status/SYSTEM_STATUS.md)** - Overall system status
- **[STATUS_REPORT.md](status/STATUS_REPORT.md)** - Detailed status report
- **[NIGHT_AGENTS_STATUS.md](status/NIGHT_AGENTS_STATUS.md)** - Overnight agent status
- **[SLEEP_WELL.md](status/SLEEP_WELL.md)** - Night agent logs

---

## Config Templates

TOML configuration templates:

- **[routing.toml](config/routing.toml)** - MoE routing configuration
- **[model_registry.toml](config/model_registry.toml)** - Model registry template

---

## Backlog

TODOs and improvement tracking:

- **[IMPROVEMENTS.md](backlog/IMPROVEMENTS.md)** - General improvements
- **[KNOWLEDGE_BASE_TODOS.md](backlog/KNOWLEDGE_BASE_TODOS.md)** - Knowledge base tasks
- **[KB_TODO_ITEMS_1_2_3.md](backlog/KB_TODO_ITEMS_1_2_3.md)** - Specific KB items

---

## Root-Level Docs

Miscellaneous documentation:

- **[DISTRIBUTED_CAMPAIGN_READY.md](DISTRIBUTED_CAMPAIGN_READY.md)** - Distributed training campaign readiness
- **[cloud_testing_setup.md](cloud_testing_setup.md)** - Cloud testing setup
- **[llamacpp_usage.md](llamacpp_usage.md)** - llama.cpp usage guide
- **[training_monitoring.md](training_monitoring.md)** - Training monitoring

---

## Navigation Tips

**By Topic**:
- **Getting Started**: See [Quick Start](#quick-start)
- **Cognitive Protocol**: [architecture/COGNITIVE_PROTOCOL.md](architecture/COGNITIVE_PROTOCOL.md), [guides/COGNITIVE_CONFIG_GUIDE.md](guides/COGNITIVE_CONFIG_GUIDE.md)
- **Model Training**: See [Training](#training) section, especially [ORACLE_FARORE_SECRETS_PLAN.md](training/ORACLE_FARORE_SECRETS_PLAN.md)
- **Multi-Agent System**: [guides/AGENTIC_SYSTEM_GUIDE.md](guides/AGENTIC_SYSTEM_GUIDE.md), [architecture/MOE_SYSTEM.md](architecture/MOE_SYSTEM.md)
- **Windows Setup**: See [Windows Setup](#windows-setup)

**By Recency**:
- **Latest (2025-12-22)**: [Recent Additions](#recent-additions-2025-12-22)
- **Previous Day (2025-12-21)**: MOE_SYSTEM.md, various training docs

**By Status**:
- **Current/Active**: Docs in [Recent Additions](#recent-additions-2025-12-22), [Training](#training), [Architecture](#architecture)
- **Planning/Future**: [Plans](#plans)
- **Historical/Reference**: [protocols/COGNITIVE_PROTOCOL.md](protocols/COGNITIVE_PROTOCOL.md) (older version, see architecture/ for latest)

---

**Need help?** Check the [Quick Start](#quick-start) or the [Guides](#guides) section.
