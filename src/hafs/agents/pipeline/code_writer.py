from hafs.agents.base import BaseAgent

class CodeWriter(BaseAgent):
    """Writes code for a single file based on a plan."""
    def __init__(self):
        super().__init__("CodeWriter", "Writes code for a single file based on a plan.")
        self.model_tier = "coding"

    async def run_task(self, file_path: str, instructions: str, tdd: str) -> str:
        prompt = f"File: {file_path}\nInstructions: {instructions}\nTDD:\n{tdd}\n\nWrite the code for this file. Output ONLY the raw code."
        return await self.generate_thought(prompt)
