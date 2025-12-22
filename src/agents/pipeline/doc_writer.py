from agents.swarm.swarm.specialists import DeepDiveDocumenter

class DocWriter(DeepDiveDocumenter):
    """Generates a Technical Design Doc (TDD) from a dev prompt."""
    def __init__(self):
        super().__init__()
        self.name = "DocWriter"
        self.role_description = "Generates a Technical Design Doc (TDD) from a dev prompt."

    async def run_task(self, dev_prompt: str) -> str:
        prompt = f"Use the following dev prompt to write a detailed TDD. Infer the architecture and components.\n\nPROMPT:\n{dev_prompt}"
        return await self.generate_thought(prompt)
