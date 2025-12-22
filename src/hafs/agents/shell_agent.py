import warnings
from agents.utility.shell_agent import ShellAgent

warnings.warn(
    "agents.utility.shell_agent is deprecated. Import from 'agents.utility.shell_agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ShellAgent"]
