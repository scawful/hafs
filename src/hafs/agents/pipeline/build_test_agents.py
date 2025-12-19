from hafs.agents.shell_agent import ShellAgent

class BuildAgent(ShellAgent):
    """Generic Build Agent. Runs a build command (e.g. 'make', 'npm build')."""
    def __init__(self, workspace_path: str):
        super().__init__(workspace_path, execution_mode="build_only")
        self.name = "BuildAgent"

class TestAgent(ShellAgent):
    """Generic Test Agent. Runs a test command (e.g. 'pytest', 'npm test')."""
    def __init__(self, workspace_path: str):
        super().__init__(workspace_path, execution_mode="build_only")
        self.name = "TestAgent"
