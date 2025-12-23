# hAFS Infrastructure Roadmap

This document outlines the active plans and future improvements for the hAFS infrastructure.

## 1. Local AI Orchestration
**Status:** 70% Complete
**Goal:** Enable intelligent local AI across all hafs agents with priority-based inference.

- [ ] Implement `LocalAIOrchestrator` with priority queuing.
- [ ] Integrate with `consolidation_analyzer.py` for AI-powered filesystem recommendations.
- [ ] Implement MoE (Mixture of Experts) orchestrator to route complex tasks.
- [ ] Deploy and test on Windows/GPU nodes.

## 2. Training Data Quality
**Goal:** Increase the rate of high-quality samples (>0.6 quality score).

- [ ] Analyze "Golden Templates" (e.g., the 0.76 hardware reset routine).
- [ ] Refine quality component weighting (diversity, KG, hallucination, coherence).
- [ ] Implement source filtering to prioritize high-quality codebanks.
- [ ] Implement checkpoint validation and auto-rollback for training resilience.

## 3. Visualization & Monitoring
**Goal:** Real-time visibility into training, quality, and filesystem status.

- [ ] Add live sample generation metrics to the C++/ImGui dashboard.
- [ ] Visualize hybrid orchestrator decisions (GPU vs API).
- [ ] Implement an interactive "Quality Inspector" for sample review.

## 4. Knowledge Graph & Semantic Search
**Goal:** Enable deep retrieval across multiple codebases and systems.

- [ ] Generate embeddings for global filesystem scans.
- [ ] Link ALTTP routines to Gigaleak equivalents.
- [ ] Implement cross-codebase semantic search for common patterns.

## 5. Multi-Machine Optimization
**Goal:** Seamless workflow between Mac (Orchestrator) and Windows (Training).

- [ ] Automate sync scripts (`rsync`) for datasets and configs.
- [ ] Integrate Tailscale for automatic mounts across locations.
- [ ] Implement daily backups of knowledge graphs and datasets.

---

*Last Updated: 2025-12-22*
