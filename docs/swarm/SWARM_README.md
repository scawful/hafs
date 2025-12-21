# Swarm System - Quick Reference

The swarm system is now fully implemented for YAZE and Mesen2 analysis.

## Quick Launch

```bash
# Single missions
python -m agents.swarm.launcher yaze-performance
python -m agents.swarm.launcher yaze-audio
python -m agents.swarm.launcher yaze-input
python -m agents.swarm.launcher mesen2-integration

# All YAZE missions
python -m agents.swarm.launcher yaze-all

# Everything (YAZE + Mesen2)
python -m agents.swarm.launcher all
```

## Documentation

- **Mission Specs**: `docs/swarm/SWARM_MISSIONS.md`
- **Usage Guide**: `docs/swarm/SWARM_USAGE.md`
- **MoE System**: `docs/architecture/MOE_SYSTEM.md`

## Implementation

- **YAZE Specialists**: `src/agents/swarm/yaze_specialists.py`
- **YAZE Coordinator**: `src/agents/swarm/yaze_swarm.py`
- **Mesen2 Specialists**: `src/agents/swarm/mesen2_specialists.py`
- **Mesen2 Coordinator**: `src/agents/swarm/mesen2_swarm.py`
- **Launcher CLI**: `src/agents/swarm/launcher.py`

## Outputs

- **YAZE Reports**: `~/.context/swarms/yaze/`
- **Mesen2 Reports**: `~/.context/swarms/mesen2/`
- **Lua Scripts**: `~/.context/swarms/mesen2/lua_scripts/`

---

**Status**: ✓ READY TO USE
**Created**: 2025-12-21
