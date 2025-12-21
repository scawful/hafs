"""Visualizer Agent (Public Port).

Generates Mermaid.js diagrams from text descriptions.
"""

from agents.core.base import BaseAgent

class VisualizerAgent(BaseAgent):
    """The Artist. Visualizes systems."""

    def __init__(self):
        super().__init__("Visualizer", "Generate diagrams/charts from context.")

    async def create_diagram(self, context: str, focus: str = "System") -> dict:
        prompt = (
            f"Create a Mermaid.js diagram representing the following system context.\n"
            f"FOCUS: {focus}\n\n"
            f"CONTEXT:\n{context[:5000]}\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the mermaid code block."
        )
        
        content = await self.generate_thought(prompt)
        
        # Strip markdown
        if "```mermaid" in content:
            content = content.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return {"type": "mermaid", "content": content}

