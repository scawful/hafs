# HAFS Agents Module - Quick Start

## Installation & Import

The agents module is located at `~/Code/hafs/src/hafs/agents/`

```python
from hafs.agents import (
    AgentCoordinator,
    AgentLane,
    MentionRouter,
    AgentRole,
)
```

## Quick Examples

### 1. Create and Configure Coordinator

```python
coordinator = AgentCoordinator({
    "max_agents": 5,
    "default_backend": "claude",
})
```

### 2. Register Agents

```python
await coordinator.register_agent(
    name="planner",
    role=AgentRole.PLANNER,
    backend_name="claude",
    system_prompt="You are a planning specialist."
)
```

### 3. Route Messages

```python
# With @mention
recipient = await coordinator.route_message("@planner create a roadmap")

# Auto-routing by content
recipient = await coordinator.route_message("implement the login function")
```

### 4. Stream Responses

```python
async for chunk in coordinator.stream_agent_response(recipient):
    print(chunk, end="")
```

### 5. Shared Context

```python
coordinator.update_shared_context(
    task="Build API",
    finding="Database ready",
    decision="Use REST"
)
```

### 6. Broadcast

```python
await coordinator.broadcast("Deadline changed!", sender="system")
```

## Agent Roles & Keywords

| Role | Keywords |
|------|----------|
| PLANNER | plan, strategy, organize, roadmap, steps |
| CODER | code, implement, write, create, build, develop |
| CRITIC | review, check, verify, validate, improve |
| RESEARCHER | research, investigate, analyze, explore, find |
| GENERAL | help, assist, general, question, explain |

## Module Structure

```
hafs/agents/
├── __init__.py          # Package exports
├── coordinator.py       # AgentCoordinator class
├── lane.py             # AgentLane & AgentLaneManager
├── roles.py            # Role definitions & utilities
├── router.py           # MentionRouter class
└── README.md           # Full documentation

hafs/models/
└── agent.py            # Agent, AgentMessage, AgentRole, SharedContext
```

## Testing

```bash
# Test imports
python3 test_agents_import.py

# Run examples
python3 examples/multi_agent_example.py
```

## API Quick Reference

### AgentCoordinator

| Method | Description |
|--------|-------------|
| `register_agent(name, role, backend_name, system_prompt)` | Register new agent |
| `unregister_agent(name)` | Remove agent |
| `route_message(message, sender)` | Route message to agent |
| `broadcast(message, sender)` | Send to all agents |
| `update_shared_context(task, finding, decision)` | Update shared state |
| `stream_agent_response(agent_name)` | Stream agent output |
| `start_all_agents()` | Start all agents |
| `stop_all_agents()` | Stop all agents |

### AgentLane

| Method | Description |
|--------|-------------|
| `start()` | Start the lane |
| `stop()` | Stop the lane |
| `receive_message(msg)` | Queue message |
| `process_next_message()` | Process queued message |
| `stream_output()` | Stream response |

### MentionRouter

| Method | Description |
|--------|-------------|
| `extract_mentions(message)` | Get @mentions |
| `strip_mentions(message)` | Remove @mentions |
| `route_by_content(message, agents)` | Infer recipient |
| `resolve_recipient(message, agents)` | Full routing logic |

### SharedContext

| Method | Description |
|--------|-------------|
| `add_finding(finding)` | Add key finding |
| `add_decision(decision)` | Add decision |
| `to_prompt_text()` | Format for prompts |

## Configuration Options

```python
config = {
    "max_agents": 10,              # Max concurrent agents
    "default_backend": "claude",   # Default AI backend
    "enable_context_sharing": True # Enable shared context
}
```

## File Locations

- **Module**: `~/Code/hafs/src/hafs/agents/`
- **Models**: `~/Code/hafs/src/hafs/models/agent.py`
- **Tests**: `~/Code/hafs/test_agents_import.py`
- **Examples**: `~/Code/hafs/examples/multi_agent_example.py`
- **Docs**: `~/Code/hafs/src/hafs/agents/README.md`

## Next Steps

1. Read the full documentation: `src/hafs/agents/README.md`
2. Run the test suite: `python3 test_agents_import.py`
3. Try the examples: `python3 examples/multi_agent_example.py`
4. Integrate with your HAFS workflows!
