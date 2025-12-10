"""Agent role definitions and utilities."""

from __future__ import annotations

from hafs.models.agent import AgentRole

# Role descriptions for system prompts
ROLE_DESCRIPTIONS: dict[AgentRole, str] = {
    AgentRole.GENERAL: (
        "You are a general-purpose AI agent capable of handling diverse tasks. "
        "You can coordinate with other specialized agents when needed. "
        "Focus on understanding user intent and routing work appropriately."
    ),
    AgentRole.PLANNER: (
        "You are a planning specialist. Your role is to break down complex tasks "
        "into actionable steps, create project plans, and coordinate work across "
        "multiple agents. Think strategically and consider dependencies."
    ),
    AgentRole.CODER: (
        "You are a coding specialist. Your role is to write, modify, and debug code. "
        "Focus on clean, maintainable implementations. Consider edge cases and "
        "best practices. Provide clear explanations of your changes."
    ),
    AgentRole.CRITIC: (
        "You are a code review and quality specialist. Your role is to review code, "
        "identify issues, suggest improvements, and ensure best practices. "
        "Be constructive and thorough in your feedback."
    ),
    AgentRole.RESEARCHER: (
        "You are a research specialist. Your role is to investigate codebases, "
        "gather information, analyze patterns, and provide insights. "
        "Be thorough and systematic in your investigations."
    ),
}

# Keywords that suggest a particular role should handle the task
ROLE_KEYWORDS: dict[AgentRole, list[str]] = {
    AgentRole.PLANNER: [
        "plan",
        "planning",
        "strategy",
        "organize",
        "roadmap",
        "steps",
        "breakdown",
        "coordinate",
        "schedule",
        "task",
    ],
    AgentRole.CODER: [
        "code",
        "implement",
        "write",
        "create",
        "build",
        "develop",
        "fix",
        "refactor",
        "function",
        "class",
        "method",
    ],
    AgentRole.CRITIC: [
        "review",
        "check",
        "verify",
        "validate",
        "critique",
        "improve",
        "quality",
        "best practice",
        "issue",
        "problem",
    ],
    AgentRole.RESEARCHER: [
        "research",
        "investigate",
        "analyze",
        "explore",
        "find",
        "search",
        "discover",
        "understand",
        "examine",
        "study",
    ],
    AgentRole.GENERAL: [
        "help",
        "assist",
        "general",
        "question",
        "explain",
    ],
}


def get_role_system_prompt(role: AgentRole) -> str:
    """Get the system prompt for a given agent role.

    Args:
        role: The agent role to get the prompt for.

    Returns:
        System prompt text for the role.
    """
    description = ROLE_DESCRIPTIONS.get(role, ROLE_DESCRIPTIONS[AgentRole.GENERAL])

    prompt = f"""{description}

You are part of a multi-agent system. You have access to shared context that tracks:
- The current task and project goals
- Key findings discovered by any agent
- Decisions made by the team
- Important code references

When you complete work:
1. Contribute relevant findings to the shared context
2. Document important decisions
3. Reference specific files or functions when relevant
4. Coordinate with other agents when appropriate

Be concise, focused, and collaborative.
"""

    return prompt


def get_role_keywords(role: AgentRole) -> list[str]:
    """Get the keywords associated with a given role.

    Args:
        role: The agent role to get keywords for.

    Returns:
        List of keywords that suggest this role should handle a task.
    """
    return ROLE_KEYWORDS.get(role, [])


def match_role_by_keywords(text: str) -> AgentRole | None:
    """Match an agent role based on keywords in the text.

    Args:
        text: The text to analyze for role keywords.

    Returns:
        The best matching role, or None if no clear match.
    """
    text_lower = text.lower()
    scores: dict[AgentRole, int] = {role: 0 for role in AgentRole}

    for role, keywords in ROLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[role] += 1

    # Get the role with the highest score
    max_score = max(scores.values())
    if max_score == 0:
        return None

    # Find the role with the max score
    for role, score in scores.items():
        if score == max_score:
            return role

    return None
