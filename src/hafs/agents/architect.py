"""Architect Agent (Public Port)."""

from agents.core.base import BaseAgent

class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("Architect", "Draft design documents.")

    async def run_task(self, feature_request: str) -> str:
        prompt = (
            f"Write a Design Doc for: {feature_request}\n"
            "Structure: Objective, Background, Design, Alternatives."
        )
        doc = await self.generate_thought(prompt)
        
        # Save
        name = feature_request.split()[0]
        (self.knowledge_dir / "discovered" / f"design_{name}.md").write_text(doc)
        return doc

