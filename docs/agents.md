# HAFS Multi-Agent Orchestration System

The agents module provides a comprehensive multi-agent orchestration system for coordinating multiple AI agents working together on complex tasks.

## Architecture

The module is organized into five main components:

### 1. Models (`hafs.models.agent`)

Core data models for the multi-agent system:

- **AgentRole**: Enum defining agent roles (GENERAL, PLANNER, CODER, CRITIC, RESEARCHER)
- **Agent**: Represents an agent with a role, backend, and system prompt
- **AgentMessage**: Messages exchanged between agents
- **SharedContext**: Shared state accessible to all agents

### 2. Roles (`hafs.agents.roles`)

Role definitions and utilities:

- **ROLE_DESCRIPTIONS**: Detailed descriptions for each role
- **ROLE_KEYWORDS**: Keywords that suggest which role should handle a task
- **get_role_system_prompt(role)**: Generate system prompts for roles
- **match_role_by_keywords(text)**: Infer the best role based on content

### 3. Router (`hafs.agents.router`)

Message routing with mention support:

- **MentionRouter**: Routes messages using @mentions or content analysis
  - `extract_mentions(message)`: Extract @mentions from text
  - `strip_mentions(message)`: Remove @mentions from text
  - `route_by_content(message, agents)`: Infer recipient from keywords
  - `resolve_recipient(message, agents)`: Resolve recipient and clean message

### 4. Lane (`hafs.agents.lane`)

Individual agent execution contexts:

- **AgentLane**: Manages a single agent's backend and message queue
  - `start()`, `stop()`: Lifecycle management
  - `receive_message(msg)`: Queue messages for processing
  - `stream_output()`: Stream agent responses
  - `_build_context_prompt(msg)`: Inject shared context into prompts

- **AgentLaneManager**: Utility for managing multiple lanes

### 5. Coordinator (`hafs.agents.coordinator`)

High-level orchestration:

- **AgentCoordinator**: Coordinates multiple agents
  - `register_agent()`: Add agents to the system
  - `route_message()`: Route messages using @mentions or content
  - `broadcast()`: Send messages to all agents
  - `update_shared_context()`: Update context accessible to all agents
  - `stream_agent_response()`: Stream responses from specific agents

## Usage Examples

### Basic Setup

```python
from hafs.agents import AgentCoordinator, AgentRole

# Create coordinator
coordinator = AgentCoordinator({
    "max_agents": 5,
    "default_backend": "claude",
})

# Register agents
await coordinator.register_agent(
    name="alice",
    role=AgentRole.PLANNER,
    system_prompt="You are a planning specialist..."
)

await coordinator.register_agent(
    name="bob",
    role=AgentRole.CODER,
    system_prompt="You are a coding specialist..."
)

# Start all agents
await coordinator.start_all_agents()
```

### Routing Messages

```python
# Explicit routing with @mention
recipient = await coordinator.route_message(
    "@alice create a project roadmap"
)
async for chunk in coordinator.stream_agent_response(recipient):
    print(chunk, end="")

# Content-based routing (automatically routes to coder)
recipient = await coordinator.route_message(
    "implement the login function"
)
async for chunk in coordinator.stream_agent_response(recipient):
    print(chunk, end="")
```

### Shared Context

```python
# Update shared context
coordinator.update_shared_context(
    task="Build authentication system",
    finding="User model exists in database",
    decision="Using JWT for session management"
)

# Context is automatically injected into all agent prompts
recipient = await coordinator.route_message(
    "@bob implement the JWT token generation"
)
```

### Broadcasting

```python
# Broadcast to all agents
recipients = await coordinator.broadcast(
    "Project goal updated: Deploy by end of week",
    sender="system"
)
```

### Direct Lane Management

```python
from hafs.agents.lane import AgentLane
from hafs.backends import BackendRegistry
from hafs.models.agent import Agent, AgentRole, SharedContext, AgentMessage

# Create components
agent = Agent(
    name="planner",
    role=AgentRole.PLANNER,
    backend_name="claude",
    system_prompt="You are a planning specialist..."
)
backend = BackendRegistry.get("claude")
context = SharedContext(active_task="Plan the project")

# Create and start lane
lane = AgentLane(agent, backend, context)
await lane.start()

# Send message
message = AgentMessage(
    content="Create a 5-step roadmap",
    sender="user",
    recipient="planner"
)
await lane.receive_message(message)

# Process and stream output
await lane.process_next_message()
async for chunk in lane.stream_output():
    print(chunk, end="")

await lane.stop()
```

## Agent Roles

### GENERAL
General-purpose agent for diverse tasks and coordination.

**Keywords**: help, assist, general, question, explain

### PLANNER
Strategic planning and task breakdown specialist.

**Keywords**: plan, planning, strategy, organize, roadmap, steps, breakdown, coordinate, schedule, task

### CODER
Code implementation and development specialist.

**Keywords**: code, implement, write, create, build, develop, fix, refactor, function, class, method

### CRITIC
Code review and quality assurance specialist.

**Keywords**: review, check, verify, validate, critique, improve, quality, best practice, issue, problem

### RESEARCHER
Investigation and analysis specialist.

**Keywords**: research, investigate, analyze, explore, find, search, discover, understand, examine, study

## Message Routing

The MentionRouter supports two routing modes:

## Background Agents & Projects

The Autonomous Context Agent reads the project catalog defined in `hafs.toml` and
uses tool profiles to determine which commands it can run for each repo. This
lets it build per-project inventory snapshots and search results without
overstepping tool permissions.

1. **Explicit Routing**: Use @mentions to specify recipient
   ```python
   "@alice create a plan"  # Routes to agent named "alice"
   "@planner help me"      # Routes to first agent with PLANNER role
   ```

2. **Content-Based Routing**: Automatically infers recipient from keywords
   ```python
   "implement the function"    # Routes to CODER
   "review this code"          # Routes to CRITIC
   "what's in the codebase?"   # Routes to RESEARCHER
   ```

## Shared Context

All agents have access to shared context containing:

- **active_task**: Current task being worked on
- **plan**: List of plan steps
- **findings**: Key discoveries (max 50, FIFO)
- **decisions**: Important decisions (max 20, FIFO)

Context is automatically formatted and injected into agent prompts:

```
=== Shared Context ===

Active Task: Build authentication system

Plan:
  1. Design user model
  2. Implement JWT tokens
  3. Create login endpoint

Key Findings:
  - User model exists in database
  - bcrypt library already installed

Decisions Made:
  - Using JWT for session management
  - Storing tokens in HTTP-only cookies

=== End Shared Context ===
```

## Configuration

AgentCoordinator accepts a configuration dictionary:

```python
config = {
    "max_agents": 10,              # Maximum number of agents
    "default_backend": "claude",   # Default backend for agents
    "enable_context_sharing": True # Enable shared context
}
```

## Error Handling

The system includes comprehensive error handling:

```python
try:
    await coordinator.register_agent(
        name="duplicate",
        role=AgentRole.GENERAL
    )
except ValueError as e:
    print(f"Registration failed: {e}")

try:
    recipient = await coordinator.route_message(
        "some message with no suitable agent"
    )
except ValueError as e:
    print(f"Routing failed: {e}")
```

## Integration with Backends

The agents module integrates with the hafs backend system:

```python
from hafs.backends import BackendRegistry, ClaudeCliBackend

# Backends are automatically retrieved during agent registration
# The coordinator uses BackendRegistry.create() to instantiate backends

# You can use any registered backend
await coordinator.register_agent(
    name="agent1",
    role=AgentRole.CODER,
    backend_name="claude"  # Uses ClaudeCliBackend
)

await coordinator.register_agent(
    name="agent2",
    role=AgentRole.RESEARCHER,
    backend_name="gemini"  # Uses GeminiCliBackend
)
```

## Testing

See `test_agents_import.py` in the project root for import verification and basic functionality tests.

## API Reference

For detailed API documentation, see the docstrings in each module:

- `hafs.models.agent`: Core data models
- `hafs.agents.roles`: Role definitions and utilities
- `hafs.agents.router`: Message routing
- `hafs.agents.lane`: Agent execution contexts
- `hafs.agents.coordinator`: High-level orchestration
