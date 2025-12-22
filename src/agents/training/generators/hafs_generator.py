"""HAFS System Data Generator.

Generates instruction-tuning data for the HAFS CLI, scripts, and context tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.loader import load_config
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class HafsSourceItem(SourceItem):
    """Source item for HAFS system tools."""
    
    command: str = ""
    description: str = ""
    category: str = ""  # alias, script, concept
    usage_example: str = ""

    @property
    def item_id(self) -> str:
        if self.category == "cli" and self.command:
            return f"hafs:{self.category}:{self.command}"
        return f"hafs:{self.category}:{self.name}"


class HafsSystemGenerator(DataGenerator):
    """Generate instruction-tuning data for HAFS usage. 
    
    Covers:
    1. CLI Aliases (hc, htw, etc.)
    2. Maintenance Scripts (sync, deploy)
    3. Context Management (ctx usage)
    """

    def __init__(self):
        super().__init__(
            name="HafsSystemGenerator",
            domain="tool-use",
            teacher_tier="coding",
        )
        self._orchestrator = None

    async def setup(self):
        await super().setup()
        from core.orchestrator_v2 import UnifiedOrchestrator
        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[HafsSourceItem]:
        items: list[HafsSourceItem] = []
        config = load_config()

        # 1. CLI Commands & Aliases
        # Standard commands (Headless/Automation focused)
        commands = [
            ("hafs orchestrate run \"<objective>\"", "Execute a specific task or objective in headless mode."),
            ("hafs training status", "Check the current status of training jobs (snapshot)."),
            ("hafs nodes status", "Check the health and connectivity of swarm nodes."),
            ("hafs memory recall \"<query>\"", "Search the vector database for past knowledge."),
            ("hafs context list", "List currently mounted files in the agent's context."),
            ("hafs services list", "List active background services."),
        ]

        for cmd, desc in commands:
            items.append(HafsSourceItem(
                name=cmd,
                content=f"Command: {cmd}\nDescription: {desc}",
                source="hafs-core",
                command=cmd,
                description=desc,
                category="cli",
                usage_example=cmd
            ))

        # User Aliases (from Config)
        if config.general.cli_aliases:
            for alias, cmd in config.general.cli_aliases.items():
                items.append(HafsSourceItem(
                    name=alias,
                    content=f"Alias: {alias}\nExpands to: {cmd}\nDescription: User alias for {cmd}",
                    source="hafs-alias",
                    command=cmd,
                    description=f"Alias for: {cmd}",
                    category="alias",
                    usage_example=f"{alias} ..."
                ))
        
        # 2. Key Scripts
        # TODO: Make this configurable via config.scripts
        scripts = [
            ("scripts/launch_web_hub.sh", "Launch the HAFS Streamlit web dashboard"),
            ("scripts/gather_context.sh", "Aggregates project context into a single journal entry"),
        ]

        for path, desc in scripts:
            items.append(HafsSourceItem(
                name=Path(path).name,
                content=f"Script: {path}\\nDescription: {desc}",
                source="hafs-script",
                command=path,
                description=desc,
                category="script",
                usage_example=f"./{path}"
            ))

        # 3. Concepts
        concepts = [
            ("Context Mounting", "Use `ctx mount <path>` to add files to the agent's active context window."),
            ("Memory Recall", "Use `hafs memory recall` to search the vector database for past decisions."),
            ("Agent Handoff", "Use `z3ed agent handoff` when switching tasks to ensure state is preserved."),
        ]

        for name, desc in concepts:
            items.append(HafsSourceItem(
                name=name,
                content=f"Concept: {name}\\nDescription: {desc}",
                source="hafs-concept",
                command="N/A",
                description=desc,
                category="concept",
                usage_example="See description"
            ))

        # 4. Project Awareness (Dynamic from Config)
        config = load_config()
        
        for project in config.projects:
            # Generate generic navigation command
            nav_cmd = f"cd {project.path}"
            nav_desc = f"Navigate to the {project.name} project directory."
            if project.description:
                nav_desc += f" {project.description}"
            
            items.append(HafsSourceItem(
                name=project.name,
                content=f"Project: {project.name}\\nPath: {project.path}\\nDescription: {nav_desc}",
                source="hafs-project",
                command=nav_cmd,
                description=nav_desc,
                category="project",
                usage_example=nav_cmd
            ))
            
            # Check for build script (rudimentary check)
            # We can't easily check file existence here without expanding paths properly in context of the generator running environment
            # But we can assume standard patterns or check if we are local
            try:
                expanded_path = Path(str(project.path)).expanduser()
                if (expanded_path / "build.sh").exists():
                    items.append(HafsSourceItem(
                        name=f"Build {project.name}",
                        content=f"Project: {project.name}\\nCommand: ./build.sh\\nDescription: Build the project.",
                        source="hafs-project-build",
                        command=f"cd {project.path} && ./build.sh",
                        description=f"Build the {project.name} project.",
                        category="project-build",
                        usage_example=f"cd {project.path} && ./build.sh"
                    ))
            except Exception:
                pass # Ignore path expansion errors if running in restricted env

        # 5. Knowledge & Embeddings
        knowledge_tools = [
            ("History Search", "hafs-cli history search \"query\"", "Semantic search over past session history and embeddings."),
            ("Context Search", "hafs-cli context search \"query\"", "Search for items within the currently mounted context."),
            ("Cross-Agent Memory", "hafs-cli memory cross-search \"query\"", "Search across the memories of ALL agents in the swarm."),
            ("Hyrule Historian", "mcp__hyrule-historian__search", "Search indexed ASM code and RAM documentation (Zelda specific)."),
        ]

        for name, cmd, desc in knowledge_tools:
            items.append(HafsSourceItem(
                name=name,
                content=f"Tool: {name}\\nCommand: {cmd}\\nDescription: {desc}",
                source="hafs-knowledge",
                command=cmd,
                description=desc,
                category="knowledge",
                usage_example=cmd
            ))

        logger.info(f"Extracted {len(items)} HAFS system items")
        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        if not isinstance(item, HafsSourceItem):
            raise TypeError(f"Expected HafsSourceItem, got {type(item)}")

        template = (
            "You are an expert system administrator teaching a new user how to use the HAFS (Halext Agentic File System) CLI.\\n"
            "Generate a JSON object with 3 fields:\\n"
            "1. 'instruction': A natural language query a user might ask.\\n"
            "2. 'input': The context (e.g., 'User is in /Users/scawful/Code/hafs').\\n"
            "3. 'output': The exact command or alias to solve the problem.\\n\\n"
            "ITEM DETAILS:\\n"
            "Name: {name}\\n"
            "Category: {category}\\n"
            "Command/Expansion: {command}\\n"
            "Description: {description}\\n\\n"
            "RULES:\\n"
            "- If it's an alias (e.g., 'hc'), the output MUST use the alias, not the full command.\\n"
            "- Make the instruction sound natural (e.g., 'How do I check the nodes?' vs 'Execute node status').\\n"
            "- For concepts, explain the usage in the output field.\\n\\n"
            "JSON FORMAT:\\n"
            "{{\\n"
            "  \"instruction\": \"...\",\\n"
            "  \"input\": \"...\",\\n"
            "  \"output\": \"...\"\\n"
            "}}"
        )

        return template.format(
            name=item.name,
            category=item.category,
            command=item.command,
            description=item.description
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        if not isinstance(item, HafsSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            response_content, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="coding"),
                timeout=60.0,
            )
            if not response_content:
                return None

            data = extract_json_from_response(response_content)
            if not data:
                return None

            return TrainingSample(
                instruction=str(data.get("instruction", "")),
                input=str(data.get("input", "")),
                output=str(data.get("output", "")),
                domain="tool-use",
                source=item.source,
                teacher_model=model_name,
                teacher_prompt=prompt,
                kg_entities=[item.name, item.category]
            )

        except Exception as e:
            logger.error(f"Failed to generate for {item.name}: {e}")
            return None


if __name__ == "__main__":
    async def main():
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        gen = HafsSystemGenerator()
        await gen.setup()

        # Quick test
        items = await gen.extract_source_items()
        print(f"Found {len(items)} HAFS system items")

        if items:
            # Generate for all extracted items
            result = await gen.run_generation(
                limit=len(items),
                output_path=Path("hafs_tooling_dataset.jsonl"),
            )
            print(f"Generated {result.processed} samples")
            
            # Print sample output
            import json
            if Path("test_hafs_train.jsonl").exists():
                with open("test_hafs_train.jsonl") as f:
                    print("\\nSample Output:")
                    print(json.loads(f.readline()))

    import asyncio
    asyncio.run(main())
