"""Legacy path for agent roles. Use agents.core.roles instead."""
import warnings
from agents.core.roles import (
    ROLE_DESCRIPTIONS,
    ROLE_KEYWORDS,
    get_role_keywords,
    get_role_system_prompt,
    match_role_by_keywords,
)

warnings.warn(
    "agents.core.roles is deprecated. Import from 'agents.core.roles' instead.",
    DeprecationWarning,
    stacklevel=2
)
