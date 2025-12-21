"""Mesen2 Swarm Specialists for Lua Scripting and Integration.

Specialized agents for creating Mesen2 debugging scripts and YAZE integration.
Based on SWARM_MISSIONS.md Mission 4 specifications.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.core.base import BaseAgent

logger = logging.getLogger(__name__)


class LuaScriptGeneratorAgent(BaseAgent):
    """Agent for generating Mesen2 Lua debugging scripts."""

    def __init__(self):
        super().__init__(
            "LuaScriptGenerator",
            "Generate Mesen2 Lua debugging and automation scripts"
        )

    async def run_task(self, requirements: str) -> Dict[str, Any]:
        """Generate Lua scripts for Mesen2 debugging.

        Args:
            requirements: Description of debugging needs

        Returns:
            Dict with generated scripts and usage examples
        """
        prompt = f"""Generate Mesen2 Lua debugging scripts for the following requirements:

{requirements}

Mesen2 Lua API reference:
- emu.read(addr, memType) - Read memory
- emu.write(addr, value, memType) - Write memory
- emu.addBreakpoint(addr, bpType, callback) - Add breakpoint
- emu.addEventCallback(callback, eventType) - Add event hook
- emu.log(message) - Log to console
- emu.getCycle() - Get current CPU cycle count

Memory types:
- emu.memType.snesMemory - SNES address space
- emu.memType.workRam - Work RAM
- emu.memType.saveRam - Save RAM

Breakpoint types:
- emu.breakType.exec - Execution breakpoint
- emu.breakType.read - Memory read breakpoint
- emu.breakType.write - Memory write breakpoint

Event types:
- emu.eventType.startFrame - Frame start
- emu.eventType.endFrame - Frame end

Provide output as JSON with:
{{
    "scripts": [
        {{
            "name": "watch_custom_items.lua",
            "description": "Monitor custom item slots",
            "code": "-- Lua code here...",
            "usage": "Load in Mesen2 > Script Window > Open"
        }}
    ],
    "test_scenarios": [
        {{
            "script": "watch_custom_items.lua",
            "rom": "alttp_custom.sfc",
            "expected_output": "Item slot $00 changed: $00 → $01"
        }}
    ]
}}

Output ONLY valid JSON."""

        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"Failed to parse Lua script output: {e}")

        return {
            "scripts": [],
            "test_scenarios": []
        }


class IntegrationArchitectAgent(BaseAgent):
    """Agent for designing YAZE ↔ Mesen2 integration architecture."""

    def __init__(self):
        super().__init__(
            "IntegrationArchitect",
            "Design YAZE and Mesen2 integration strategy"
        )

    async def run_task(self, context: str) -> Dict[str, Any]:
        """Design integration architecture between YAZE and Mesen2.

        Args:
            context: Integration requirements and constraints

        Returns:
            Dict with architecture design and recommendations
        """
        prompt = f"""Design an integration architecture between YAZE ROM editor and Mesen2 emulator.

Context:
{context}

Consider two approaches:
1. **Plugin Architecture**: Mesen2 plugin that interfaces with YAZE
   - Pros: No forking, easier maintenance, upstream updates
   - Cons: Limited by plugin API capabilities

2. **Fork with Custom Features**: Custom Mesen2 fork with YAZE integration
   - Pros: Full control, tight integration, custom features
   - Cons: Maintenance burden, must sync upstream

For the recommended approach, provide:
- Architecture diagram (as text/ASCII)
- Communication protocol (shared memory, IPC, etc.)
- API design
- Implementation roadmap

Provide output as JSON with:
{{
    "recommendation": "plugin|fork",
    "rationale": "...",
    "architecture": {{
        "components": [
            {{"name": "YAZE Interface", "responsibility": "...", "technology": "C++"}}
        ],
        "communication": {{"method": "shared_memory|ipc|sockets", "protocol": "..."}},
        "data_flow": "YAZE → Interface → Mesen2 → Display"
    }},
    "implementation_steps": [
        {{"phase": 1, "task": "Create shared memory interface", "duration": "1 week"}}
    ],
    "risks": [
        {{"risk": "Platform compatibility", "mitigation": "...", "severity": "medium"}}
    ]
}}

Output ONLY valid JSON."""

        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"Failed to parse integration architect output: {e}")

        return {
            "recommendation": "plugin",
            "rationale": "Unknown",
            "architecture": {},
            "implementation_steps": [],
            "risks": []
        }


class DebuggingToolsBuilderAgent(BaseAgent):
    """Agent for building debugging tools integration."""

    def __init__(self):
        super().__init__(
            "DebuggingToolsBuilder",
            "Design debugging tools for YAZE-Mesen2 integration"
        )

    async def run_task(self, requirements: str) -> Dict[str, Any]:
        """Design debugging tools for integrated workflow.

        Args:
            requirements: Debugging tool requirements

        Returns:
            Dict with tool designs and implementations
        """
        prompt = f"""Design debugging tools for integrated YAZE-Mesen2 workflow.

Requirements:
{requirements}

Tools to design:
1. Memory inspector integration - Sync memory view between YAZE and Mesen2
2. Disassembly viewer sync - Show same code location in both tools
3. Breakpoint manager - Share breakpoints between tools
4. Watch expression system - Monitor values across both tools

For each tool, provide:
- Feature specification
- User workflow
- Technical implementation
- API requirements

Provide output as JSON with:
{{
    "tools": [
        {{
            "name": "Memory Inspector Sync",
            "features": ["real-time sync", "bidirectional updates"],
            "workflow": "User clicks address in YAZE → Mesen2 jumps to same address",
            "implementation": "Shared memory map with event listeners",
            "api_needed": ["yaze.onMemoryView(addr)", "mesen.jumpToAddress(addr)"]
        }}
    ],
    "integration_points": [
        {{"component": "YAZE memory viewer", "hook": "onAddressSelect", "action": "notify Mesen2"}}
    ]
}}

Output ONLY valid JSON."""

        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"Failed to parse debugging tools output: {e}")

        return {
            "tools": [],
            "integration_points": []
        }


class TestAutomationAgent(BaseAgent):
    """Agent for creating automated ROM testing framework."""

    def __init__(self):
        super().__init__(
            "TestAutomation",
            "Create automated testing framework for ROM hacks"
        )

    async def run_task(self, testing_needs: str) -> Dict[str, Any]:
        """Design automated testing framework.

        Args:
            testing_needs: Description of testing requirements

        Returns:
            Dict with test framework design
        """
        prompt = f"""Design an automated testing framework for ALTTP ROM hacks using Mesen2.

Testing needs:
{testing_needs}

Framework should support:
1. Automated ROM testing - Load ROM, run test scenarios, validate
2. Regression tests - Detect when changes break existing functionality
3. Performance benchmarks - Measure FPS, load times, etc.
4. Input playback tests - Record and replay input sequences

Provide output as JSON with:
{{
    "test_framework": {{
        "name": "ALTTP Test Suite",
        "architecture": "Mesen2 Lua + Python orchestration",
        "components": ["test_runner.lua", "rom_validator.py", "benchmark.lua"]
    }},
    "test_types": [
        {{
            "type": "regression",
            "description": "Validate core game mechanics",
            "example_tests": ["Link can move", "Items can be picked up"],
            "implementation": "Lua script that checks game state"
        }}
    ],
    "automation_scripts": [
        {{
            "script": "run_regression_suite.py",
            "description": "Run all regression tests",
            "usage": "python run_regression_suite.py --rom alttp.sfc"
        }}
    ]
}}

Output ONLY valid JSON."""

        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"Failed to parse test automation output: {e}")

        return {
            "test_framework": {},
            "test_types": [],
            "automation_scripts": []
        }


class LuaScriptLibraryGenerator(BaseAgent):
    """Agent for generating comprehensive Lua script library."""

    def __init__(self):
        super().__init__(
            "LuaLibraryGenerator",
            "Generate comprehensive Mesen2 Lua script library"
        )

    async def run_task(self, script_categories: str) -> Dict[str, Any]:
        """Generate library of Lua scripts for common ROM hacking tasks.

        Args:
            script_categories: Categories of scripts to generate

        Returns:
            Dict with complete script library
        """
        prompt = f"""Generate a comprehensive library of Mesen2 Lua scripts for ALTTP ROM hacking.

Script categories:
{script_categories}

For each category, generate 2-3 useful scripts with:
- Clear documentation
- Usage examples
- Configuration options
- Error handling

Categories to cover:
1. Memory Watch - Monitor game state, item slots, flags
2. Performance Profiling - Track CPU cycles, routine calls
3. Event Logging - Log game events (item pickups, door transitions)
4. Input Recording - Record and playback input sequences
5. Automated Testing - Test ROM hack features

Provide output as JSON with:
{{
    "library": [
        {{
            "category": "Memory Watch",
            "scripts": [
                {{
                    "name": "watch_custom_items.lua",
                    "description": "Monitor custom item slots for changes",
                    "code": "-- Full Lua code here",
                    "configuration": {{"ITEM_START": "0x7EF340", "ITEM_COUNT": 16}},
                    "usage": "Load script, custom items will be logged on change"
                }}
            ]
        }}
    ],
    "quick_start_guide": "1. Open Mesen2\\n2. Tools > Script Window\\n3. Load script..."
}}

Output ONLY valid JSON."""

        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"Failed to parse Lua library output: {e}")

        return {
            "library": [],
            "quick_start_guide": ""
        }
