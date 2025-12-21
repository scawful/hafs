from agents.core.base import BaseAgent

class TestWriter(BaseAgent):
    """Writes unit/integration tests."""
    def __init__(self):
        super().__init__("TestWriter", "Writes unit/integration tests.")
        self.model_tier = "coding"

    async def run_task(self, file_path: str, acceptance_criteria: str) -> str:
        prompt = f"File to test: {file_path}\nAcceptance Criteria: {acceptance_criteria}\n\nWrite the tests. Output ONLY the raw code."
        return await self.generate_thought(prompt)
