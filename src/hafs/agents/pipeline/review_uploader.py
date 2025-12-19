from hafs.agents.shell_agent import ShellAgent

class ReviewUploader(ShellAgent):
    """Generic Review Uploader. Creates a PR/CL/Commit."""
    
    def __init__(self, workspace_path: str):
        super().__init__(workspace_path)
        self.name = "ReviewUploader"

    async def run_task(self, description: str) -> str:
        """Uploads changes for review. Defaults to git commit."""
        # Generic implementation: Git commit
        code, out, err = await self.run_tool("git_add", args=["."])
        if code != 0:
            return f"Failed to stage changes:\n{err}"

        code, out, err = await self.run_tool("git_commit", args=[description])
        if code == 0:
            return f"Changes committed locally:\n{out}"
        return f"Failed to commit changes:\n{err}"
