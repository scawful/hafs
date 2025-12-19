"""Agent roles and keyword matching logic."""

from __future__ import annotations

from hafs.models.agent import AgentRole

ROLE_DESCRIPTIONS = {
    AgentRole.GENERAL: "A versatile assistant for various tasks.",
    AgentRole.PLANNER: "Specializes in breaking down complex tasks into manageable steps.",
    AgentRole.CODER: "Focused on writing, debugging, and explaining code.",
    AgentRole.CRITIC: "Reviews work for errors, potential issues, and improvements.",
    AgentRole.RESEARCHER: "Expert at finding information and summarizing complex topics.",
}

ROLE_KEYWORDS = {
    AgentRole.PLANNER: ["plan", "roadmap", "strategy", "break down", "task list"],
    AgentRole.CODER: ["code", "function", "class", "debug", "implement", "script", "py", "api"],
    AgentRole.CRITIC: ["review", "critique", "check", "verify", "audit", "analyze"],
    AgentRole.RESEARCHER: ["search", "find", "look up", "investigate", "what is", "how to"],
    AgentRole.GENERAL: [],
}

def get_role_keywords(role: AgentRole) -> list[str]:
    """Get keywords associated with a role."""
    return ROLE_KEYWORDS.get(role, [])

def get_role_system_prompt(role: AgentRole, persona: str | None = None) -> str:
    """Get the specific system prompt for a role."""
    if persona:
        try:
            from hafs.core.personas import PersonaRegistry

            registry = PersonaRegistry.load()
            configured = registry.get(persona)
            if configured and configured.system_prompt:
                return configured.system_prompt
        except Exception:
            pass

    try:
        from hafs.core.personas import PersonaRegistry

        registry = PersonaRegistry.load()
        configured = registry.default_for_role(role)
        if configured and configured.system_prompt:
            return configured.system_prompt
    except Exception:
        pass

    base_prompt = f"You are an AI assistant acting as a {role.value}."
    specifics = {
        AgentRole.PLANNER: (
            " Your goal is to create clear, actionable plans. "
            "Break down requests into steps."
        ),
        AgentRole.CODER: (
            " Your goal is to write clean, efficient, and well-documented code."
        ),
        AgentRole.CRITIC: (
            " Your goal is to provide constructive feedback, "
            "identifying errors and security issues."
        ),
        AgentRole.RESEARCHER: (
            " Your goal is to find accurate information and "
            "synthesize it clearly."
        ),
        AgentRole.GENERAL: " Your goal is to be helpful and versatile."
    }
    return base_prompt + specifics.get(role, "")

def match_role_by_keywords(message: str) -> AgentRole | None:
    """Infer the best agent role based on message content.

    Args:
        message: The message text to analyze.

    Returns:
        The matched AgentRole, or None if no clear match found.
    """
    message_lower = message.lower()

    # Simple keyword matching (could be improved with embeddings/LLM later)
    best_role = None
    max_matches = 0

    for role, role_keywords in ROLE_KEYWORDS.items():
        if not role_keywords:
            continue
        matches = sum(1 for keyword in role_keywords if keyword in message_lower)
        if matches > max_matches:
            max_matches = matches
            best_role = role

    return best_role
