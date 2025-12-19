"""Agent for generating development prompts from verified reports."""
from hafs.agents.base import BaseAgent
from pathlib import Path
import os
import re

class PromptEngineerAgent(BaseAgent):
    """Generates a detailed development prompt from verified knowledge."""
    
    def __init__(self):
        super().__init__("PromptEngineer", "Generates development prompts from verified reports.")
        self.model_tier = "reasoning"
        self.verified_dir = Path.home() / ".context" / "knowledge" / "verified"

    async def run_task(self, topic: str) -> str:
        """Finds a relevant report and generates a prompt."""
        
        print(f"[{self.name}] Searching for verified knowledge on '{topic}'...")
        
        # 1. Find best matching report
        best_match = None
        highest_score = -1
        
        if self.verified_dir.exists():
            for report_file in self.verified_dir.glob("*.md"):
                # Simple name matching for now
                score = 0
                for keyword in topic.lower().split():
                    if keyword in report_file.name.lower():
                        score += 1
                if score > highest_score:
                    highest_score = score
                    best_match = report_file
        
        if not best_match:
            return f"Error: No verified knowledge found for topic '{topic}'."
            
        print(f"[{self.name}] Found best match: {best_match.name}")
        
        # 2. Read the report
        report_content = best_match.read_text()
        
        # 3. Generate the prompt
        prompt = (
            "You are a Staff Engineer creating a task for a junior engineer or an AI agent.\n"
            "TASK: Convert the following 'Deep Context Report' into a clear, actionable development prompt.\n"
            "The prompt should focus on the 'Action Items' and 'Critique & Gaps' sections of the report.\n\n"
            "FORMAT:\n"
            "## Goal\n<A one-sentence summary of the task.>\n\n"
            "### Key Files\n<List the most important file paths mentioned in the report.>\n\n"
            "### Instructions\n<A numbered list of steps to take. Be specific.>\n\n"
            "### Acceptance Criteria\n<How to verify the task is complete.>\n\n"
            f"--- DEEP CONTEXT REPORT ---\n{report_content}\n"
            "--- END REPORT ---\n\n"
            "Output ONLY the generated prompt content."
        )
        
        generated_prompt = await self.generate_thought(prompt)
        
        # 4. Save to prompts directory
        prompt_dir = Path.home() / ".context" / "prompts" / "generated"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"dev_prompt_{topic.replace(' ', '_')}.md"
        prompt_path = prompt_dir / file_name
        
        prompt_path.write_text(generated_prompt)
        
        return f"Development prompt generated at: {prompt_path}"

