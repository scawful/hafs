# ALTTP Knowledge Base - Agent Integration Guide

## Quick Start

```python
# For any agent needing ALTTP knowledge
from hafs.agents.alttp_unified_kb import UnifiedALTTPKnowledge

async def get_alttp_context(query: str) -> str:
    """Get ALTTP context for a query."""
    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    results = await unified.search(query, limit=5)

    context = []
    for r in results:
        context.append(f"{r.name} ({r.kb_name}): {r.description}")
        if r.address:
            context.append(f"  Address: {r.address}")
        if r.is_hack_modification:
            context.append(f"  [MODIFIED IN HACK]")

    return "\n".join(context)
```

## Available Knowledge Bases

### 1. Vanilla ALTTP (usdasm)
- **4,851 symbols** - WRAM variables, hardware registers
- **6,591 routines** - All game routines with code
- **28 game modules** - State machine (MODE values)
- **2,200+ embeddings** - Semantic search ready

### 2. Oracle-of-Secrets (ROM hack)
- **432 symbols** - Custom WRAM, constants
- **1,269 routines** - New and modified routines
- **28 modifications** - Hooks into vanilla code
- **Bank allocations** - $20-$41 for expanded content

## Common Agent Tasks

### Task: Find Symbol Information
```python
from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase

kb = ALTTPKnowledgeBase()
await kb.setup()

# Explain a specific symbol
explanation = await kb.explain_symbol("POSX")
# Returns detailed analysis including:
# - What the symbol represents
# - How it's used in the game
# - Valid values and meanings
# - Related symbols
```

### Task: Analyze a Routine
```python
# Get deep analysis of game code
analysis = await kb.analyze_routine("Module07_Underworld")
# Returns:
# - Plain English explanation
# - Role in the game
# - Key algorithms
# - Side effects
# - How it fits in the game loop
```

### Task: Check if Something is Modified in Hack
```python
from hafs.agents.alttp_unified_kb import UnifiedALTTPKnowledge

unified = UnifiedALTTPKnowledge()
await unified.setup()

# Compare symbol between vanilla and hack
comparison = await unified.compare("SprY")

if comparison["is_modified"]:
    print("This symbol is modified in the hack")
    print(f"Vanilla: {comparison['vanilla']}")
    print(f"Hack: {comparison['hack']}")
```

### Task: Find Related Code
```python
# Semantic search for related concepts
results = await unified.search("sprite animation frame", limit=10)

for r in results:
    print(f"{r.name}: {r.description[:50]}")
    print(f"  Source: {r.kb_name}, Score: {r.score:.2f}")
```

### Task: Get Hack Modifications
```python
# Find what the hack changes
mods = await unified.get_hack_modifications()

for mod in mods:
    print(f"{mod['address']}: {mod['hack_symbol']}")
    if mod['vanilla_symbol']:
        print(f"  Overrides: {mod['vanilla_symbol']}")
    print(f"  Type: {mod['type']}")  # hook, override, extend
```

## Symbol Categories Reference

| Category | Example | Description |
|----------|---------|-------------|
| wram_direct_page | SCRAP00 | Volatile scratch variables |
| wram_link_state | POSX, POSY, LINKDO | Link's state |
| wram_sprites | SPR0_VY, SPR0_AIMODE | Sprite tables |
| wram_ancillae | ANC0MISCB | Projectiles, effects |
| register | INIDISP, BGMODE | SNES hardware |

## Key Memory Addresses

| Address | Symbol | Purpose |
|---------|--------|---------|
| $7E0010 | MODE | Current game module |
| $7E0011 | SUBMODE | Module substate |
| $7E0020 | POSY | Link Y position |
| $7E0022 | POSX | Link X position |
| $7E005D | LINKDO | Link action handler |
| $7E012C | SONG | Music track ID |
| $7E040C | DUNGEON | Current dungeon ID |

## Game Module Quick Reference

| MODE | Name | When Active |
|------|------|-------------|
| 0x07 | Underworld | In dungeons |
| 0x09 | Overworld | On world map |
| 0x0E | SaveMenu | Pause menu open |
| 0x12 | GameOver | Link died |

## Error Handling

```python
# Handle missing embeddings gracefully
results = await kb.search("some query")
if not results:
    # Embeddings may not exist for this query
    # Fall back to symbol name search
    for symbol in kb._symbols.values():
        if query.lower() in symbol.name.lower():
            results.append(symbol)
```

## Performance Tips

1. **Reuse KB instances** - Setup is expensive, reuse across calls
2. **Limit search results** - Use `limit` parameter to reduce processing
3. **Check embeddings first** - `kb._embeddings` dict shows coverage
4. **Use batch operations** - For multiple queries, batch them

## Integration Patterns

### Pattern: Context Injection
```python
async def inject_alttp_context(agent_prompt: str) -> str:
    """Inject relevant ALTTP context into agent prompt."""
    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    # Extract key terms from prompt
    terms = extract_key_terms(agent_prompt)

    # Get relevant context
    context_parts = []
    for term in terms[:3]:
        results = await unified.search(term, limit=3)
        for r in results:
            context_parts.append(f"- {r.name}: {r.description}")

    if context_parts:
        return f"""
ALTTP CONTEXT:
{chr(10).join(context_parts)}

AGENT TASK:
{agent_prompt}
"""
    return agent_prompt
```

### Pattern: Cross-Reference Check
```python
async def check_vanilla_equivalent(hack_symbol: str) -> dict:
    """Find vanilla equivalent for a hack symbol."""
    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    comparison = await unified.compare(hack_symbol)

    return {
        "exists_in_vanilla": comparison["vanilla"] is not None,
        "vanilla_info": comparison["vanilla"],
        "hack_info": comparison["hack"],
        "is_override": comparison["is_modified"],
    }
```

## Data File Locations

```
~/.context/knowledge/alttp/
├── symbols.json      # All WRAM/register symbols
├── routines.json     # All routine definitions
├── modules.json      # Game modules
├── embeddings/       # Vector embeddings
│   └── *.json       # One file per embedding
└── README.md        # Full documentation

~/.context/knowledge/oracle-of-secrets/
├── symbols.json      # Custom symbols
├── routines.json     # Custom routines
└── modifications.json # Vanilla hooks
```

## Source Code Locations

| File | Purpose |
|------|---------|
| `agents/alttp_knowledge.py` | Vanilla KB implementation |
| `agents/alttp_unified_kb.py` | Unified KB + Oracle KB |
| `agents/alttp_multi_kb.py` | Multi-source KB manager |
| `services/embedding_service.py` | Background embedding service |
| `scripts/embedding_cli.py` | CLI for embedding management |
| `scripts/test_alttp_unified_kb.py` | Test script |

## Environment Requirements

- `GEMINI_API_KEY` or `AISTUDIO_API_KEY` for embeddings
- Python 3.11+
- Source repos: ~/Code/usdasm, ~/Code/Oracle-of-Secrets
