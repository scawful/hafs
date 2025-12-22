"""Asm Instruction Generator Agent.

Extracts ALTTP assembly routines and uses a Teacher LLM to generate 
synthetic instruction tuning data for fine-tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.core.base import BaseAgent
from agents.knowledge.alttp_unified import UnifiedALTTPKnowledge
from config.prompts import get_prompt

logger = logging.getLogger(__name__)

class AsmInstructionGenerator(BaseAgent):
    """Generates synthetic instruction-tuning data from assembly sources."""

    def __init__(self):
        super().__init__(
            "AsmInstructionGenerator",
            "Generate synthetic coding instructions from assembly routines for LLM fine-tuning."
        )
        self.model_tier = "coding"  # Use coding tier (verified gemini-2.0-flash) for reliability
        self.unified_kb: Optional[UnifiedALTTPKnowledge] = None

    async def setup(self):
        """Initialize resources and knowledge bases."""
        await super().setup()
        self.unified_kb = UnifiedALTTPKnowledge()
        await self.unified_kb.setup()

    async def generate_instruction(self, routine_name: str, routine_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Use teacher model to generate an instruction for a routine."""
        
        name = routine_name
        code = routine_data.get("code", "")
        context = routine_data.get("description", "")
        memory_access = routine_data.get("memory_access", [])
        bank = routine_data.get("bank", "")

        template = get_prompt(
            "agents.analysis.asm_instruction_generator.prompt",
            "",
        )
        if not template:
            template = (
                "I will give you a valid 65816 assembly routine used in the Zelda: "
                "A Link to the Past disassembly (usdasm).\n"
                "Your task is to reverse-engineer the intent and write the user prompt "
                "(Instruction) that would request this specific code.\n\n"
                "ROUTINE NAME: {name}\n"
                "BANK: {bank}\n"
                "EXISTING DESCRIPTION: {context}\n"
                "MEMORY ACCESS: {memory_access}\n\n"
                "CODE:\n```\n{code}\n```\n\n"
                "Respond with a JSON object containing:\n"
                "1. \"instruction\": A natural language request that would lead to this code.\n"
                "2. \"input\": Any necessary context (RAM addresses, hardware registers, specific symbols).\n"
                "3. \"output\": The assembly code provided.\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            )
        prompt = template.format(
            name=name,
            bank=bank,
            context=context,
            memory_access=", ".join(memory_access),
            code=code,
        )

        try:
            from core.orchestrator_v2 import TaskTier, Provider
            
            print(f"Calling UnifiedOrchestrator for {name}...", flush=True)
            # Use the orchestrator from UnifiedKB which is already initialized (v2)
            if not self.unified_kb or not self.unified_kb._orchestrator:
                print("Error: UnifiedKB orchestrator not ready", flush=True)
                return None
            
            import asyncio
            try:
                response_obj = await asyncio.wait_for(
                    self.unified_kb._orchestrator.generate(
                        prompt=prompt,
                        tier=TaskTier.CODING,
                        provider=Provider.GEMINI
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                print(f"Timeout generating for {name}", flush=True)
                return None
                
            response = response_obj.content
            print(f"Got response for {name} ({len(response)} chars)", flush=True)

            # Extract JSON from response (simple attempt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            
            try:
                data = json.loads(response)
                return data
            except:
                print(f"JSON Parse Failed. Raw (first 200): {response[:200]}", flush=True)
                return None
        except Exception as e:
            print(f"Failed to generate instruction for {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    async def run_generation(self, output_path: Path, limit: int = 100):
        """Run the generation pipeline for a set of routines."""
        if not self.unified_kb:
            await self.setup()

        all_routines = []
        
        # Get routines from vanilla KB
        print("Loading Routines from Vanilla KB...", flush=True)
        if self.unified_kb._vanilla_kb:
             print(f"Vanilla routines count: {len(self.unified_kb._vanilla_kb._routines)}", flush=True)
             for name, routine in self.unified_kb._vanilla_kb._routines.items():
                 all_routines.append((name, routine, "vanilla"))
            
        # Get routines from hack KB
        print("Loading Routines from Hack KB...", flush=True)
        if self.unified_kb._hack_kb:
            print(f"Hack routines count: {len(self.unified_kb._hack_kb._routines)}", flush=True)
            for name, routine in self.unified_kb._hack_kb._routines.items():
                all_routines.append((name, routine, "hack"))
        
        print(f"Total routines found: {len(all_routines)}", flush=True)
        dataset = []
        count = 0
        
        for name, routine, source in all_routines:
            if count >= limit:
                break
            
            print(f"Generating for {name} ({source}) [{count+1}/{limit}]", flush=True)
            
            # routine is a dict in hack_kb and an object in vanilla_kb? 
            # Let's handle both.
            if hasattr(routine, "to_dict"):
                routine_data = routine.to_dict()
            else:
                routine_data = dict(routine)
                
            entry = await self.generate_instruction(name, routine_data)
            if entry:
                entry["source"] = source
                dataset.append(entry)
                count += 1

        # Save as JSONL
        with open(output_path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Dataset generated with {len(dataset)} entries at {output_path}")
        return len(dataset)

    async def run_task(self, task: Dict[str, Any]) -> str:
        """Run generation task."""
        output = task.get("output", "dataset.jsonl")
        limit = task.get("limit", 10)
        output_path = Path(output)
        
        count = await self.run_generation(output_path, limit)
        return f"Generated {count} instructions in {output_path}"

if __name__ == "__main__":
    # Test script
    print("Starting AsmInstructionGenerator main script...", flush=True)
    async def main():
        # Monkeypatch: Skip loading embeddings for generation task (speed optimization)
        from agents.knowledge.alttp import ALTTPKnowledgeBase
        original_load = ALTTPKnowledgeBase._load_embeddings
        ALTTPKnowledgeBase._load_embeddings = lambda self: print("Skipping embedding load for generator.", flush=True)

        print("Initializing generator...", flush=True)
        gen = AsmInstructionGenerator()
        print("Running setup...", flush=True)
        await gen.setup()
        
        # Check if KB needs building
        if not gen.unified_kb._vanilla_kb._routines:
            print("KB is empty. Building from source (no embeddings for speed)...", flush=True)
            await gen.unified_kb.build_all(generate_embeddings=False)
            print("Build complete.", flush=True)
        
        print("Starting generation...", flush=True)
        await gen.run_generation(Path("alttp_asm_train.jsonl"), limit=5)
        print("Generation finished.", flush=True)
    
    import asyncio
    asyncio.run(main())
