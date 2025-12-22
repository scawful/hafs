# ASM Embedding & Knowledge Graph Upgrade Plan

**Status:** Draft
**Context:** Improving HAFS performance on the "difficult domain" of SNES Assembly.

## Problem Statement
Current embedding pipelines treat ASM as generic text. This fails because:
1.  **Symbol Sparsity:** `LDA $10` has no semantic meaning without knowing `$10` is `GameMode`.
2.  **Context Dependence:** Register width (8-bit vs 16-bit) depends on `REP/SEP` instructions usually found far earlier in the execution flow.
3.  **Hardware Specificity:** Understanding code requires implicit knowledge of memory maps (Banks $7E/$7F) and Hardware Registers ($21xx/$42xx).

## 1. Embedding Pipeline Upgrades

### A. The "65816 Preprocessor" (Enrichment Layer)
Before embedding any routine, pass it through an enrichment layer.

*   **Symbol Injection:** Regex-replace known hex addresses with their labels *inline*.
    *   *Before:* `LDA $7EF3CA`
    *   *After:* `LDA $7EF3CA [Link_X_Coordinate]`
*   **Opcode Expansion:** (Optional) Expand obscure mnemonics to natural language descriptions.
    *   *Before:* `SEP #$30`
    *   *After:* `SEP #$30 [Set Index/Accumulator to 8-bit]`

### B. Instruction-Aware Chunking
ASM routines vary wildly in size (5 lines to 5,000 lines).
*   **Strategy:** Never split mid-instruction.
*   **Boundary Detection:** Split primarily on `RTL`, `RTS`, `RTI` (subroutine ends) or major Labels (`Module_...:`).
*   **Overlap:** Include the *preceding* 5-10 instructions in the next chunk to preserve register width context (`REP`/`SEP` status).

### C. Domain-Specific Model (Future)
Evaluate fine-tuning a small model (e.g., `bge-small-en-v1.5`) specifically on:
*   Positive pairs: (Raw ASM, Commented ASM)
*   Positive pairs: (ASM Routine, English Description)

## 2. Knowledge Graph Upgrades

### A. Deterministic Call Grapher
Do not rely on LLMs to extract `CALLS` relationships from ASM. Use a static parser.
*   **Logic:** Scan all verified code for `JSL`, `JSR`, `JML`.
*   **Resolution:** Map the target address/label to a canonical Node ID.
*   **Edge:** Create explicit `CALLS` edges in the graph.

### B. Hardware Knowledge Injection
Inject a static subgraph representing the SNES hardware.
*   **Source:** `hardware_defs.json` (derived from `fullsnes` or verified docs).
*   **Nodes:** `$2100` (INIDISP), `$4200` (NMITIMEN), etc.
*   **Edges:** When code access these addresses, create `READS_FROM` / `WRITES_TO` edges to these Hardware Nodes.
*   **Benefit:** Enables queries like "Show me all routines that modify screen brightness" (by finding parents of `$2100`).

### C. Memory Map Topology
Model the memory map as a hierarchy in the graph.
*   `WRAM ($7E0000-$7FFFFF)` -> `Link State` -> `Link X Coord`.
*   This allows "fuzzy" retrieval. A query about "Link's position" hits the parent node and traverses down to specific addresses.

## Implementation Roadmap

1.  **Prototype Preprocessor:** `src/agents/knowledge/asm_preprocessor.py`
2.  **Hardware Definitions:** Create `src/hafs/data/snes_hardware.json`.
3.  **Graph Builder Update:** Modify `KnowledgeGraphAgent.build_graph` to load the hardware defs and run the deterministic parser.
