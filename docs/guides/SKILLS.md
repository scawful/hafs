# Anthropic Skills Integration

_Last updated: 2025-12-20_

This document describes how Anthropic Skills integrate with hAFS agents.

## 1. Overview

Anthropic Skills are specialized capabilities exposed through Claude Code's Skill tool. hAFS can leverage these skills through agent role mapping.

## 2. Skill Discovery

Skills are configured in Claude settings:
```json
// ~/.claude/settings.json
{
  "enabledPlugins": {
    "frontend-design@claude-code-plugins": true
  }
}
```

Agents can discover available skills by reading this configuration.

## 3. Skill to hAFS Role Mapping

| Claude Skill | hAFS Role | Description |
|--------------|-----------|-------------|
| frontend-design | @coder | Production-grade frontend interfaces |
| web-artifacts-builder | @coder | React + Tailwind + shadcn/ui artifacts |
| project-initialization | @planner | Scaffold new projects |
| (planning) | @planner | Strategic planning and architecture |
| (review) | @critic | Quality review and analysis |
| (research) | @researcher | Investigation and knowledge gathering |

## 3.1 Recommended MCP Servers by Category

### Browser Automation
| Server | Purpose | Install |
|--------|---------|---------|
| playwright-mcp | Microsoft's browser automation, screenshots | `claude mcp add playwright-mcp -- npx -y @anthropic/mcp-playwright` |
| puppeteer | Anthropic's Chromium automation | `claude mcp add puppeteer -- npx -y @anthropic/mcp-puppeteer` |

### Design & Frontend
| Server | Purpose | Install |
|--------|---------|---------|
| figma | Design-to-code, component hierarchies | `claude mcp add figma -- npx -y figma-mcp` |

### Terminal & System
| Server | Purpose | Install |
|--------|---------|---------|
| iterm-mcp | iTerm session control, REPL | `claude mcp add iterm-mcp -- npx -y iterm-mcp` |
| desktop-commander | Full OS filesystem + terminal | `claude mcp add desktop-commander` |

### Project Management
| Server | Purpose | Install |
|--------|---------|---------|
| jira | Atlassian issues, sprints | `claude mcp add jira -- npx -y @anthropic/mcp-atlassian` |
| linear | Linear task management | `claude mcp add linear -- npx -y @anthropic/mcp-linear` |
| asana | Asana integration | `claude mcp add asana` |

### Memory & Knowledge Graph
| Server | Purpose | Install |
|--------|---------|---------|
| knowledge-graph | Anthropic's entities/relations | `claude mcp add knowledge-graph -- npx -y @anthropic/mcp-knowledge-graph` |
| memento | Neo4j + vector embeddings | `claude mcp add memento -- npx -y memento-mcp` |
| qdrant-memory | Qdrant vector database | `claude mcp add qdrant-memory` |

### Image & Multimodal
| Server | Purpose | Install |
|--------|---------|---------|
| screenshot | Website screenshots | `claude mcp add screenshot -- npx -y screenshot-mcp` |
| image-analysis | GPT-4o vision analysis | `claude mcp add image-analysis -- npx -y mcp-image-analysis` |
| glm-vision | Code screenshot analysis | `claude mcp add glm-vision` |

### Database
| Server | Purpose | Install |
|--------|---------|---------|
| postgres | PostgreSQL schema + queries | `claude mcp add postgres -- npx -y @anthropic/mcp-postgres` |
| sqlite | SQLite with built-in analysis | `claude mcp add sqlite -- npx -y @anthropic/mcp-sqlite` |
| mongodb | MongoDB collections | `claude mcp add mongodb` |

### Development
| Server | Purpose | Install |
|--------|---------|---------|
| github | Issues, PRs, actions | Already in official servers |
| sentry | Error tracking | `claude mcp add sentry` |
| git | Repository operations | Already configured |

## 3.2 Plugin Marketplaces

```bash
# Add official Anthropic marketplace
/plugin marketplace add anthropics/claude-code

# Add community marketplace (243+ plugins with Agent Skills)
/plugin marketplace add jeremylongshore/claude-code-plugins-plus

# Browse and install
/plugin menu
```

## 4. Invoking Skills from hAFS

### Via Claude Code
```python
# In Claude Code context, skills are invoked via Skill tool
Skill("frontend-design", args="create a dashboard component")
```

### Via hAFS Orchestrator
```python
from hafs.agents import AgentCoordinator, AgentRole

coordinator = AgentCoordinator()

# Register agent with Claude backend for skill access
await coordinator.register_agent(
    "ui-designer",
    AgentRole.CODER,
    backend="claude"  # Claude backend has skill access
)

# Route to skill-capable agent
response = await coordinator.route_message(
    "Create a polished dashboard with dark mode",
    target_roles=[AgentRole.CODER]
)
```

## 5. Creating Custom Skills

Skills can be added as hAFS plugins:

### Plugin Structure
```
hafs_plugins/
└── my_skill/
    ├── __init__.py
    ├── skill.py
    └── hafs_plugin.yaml
```

### Plugin Definition
```yaml
# hafs_plugin.yaml
name: my-skill
version: 1.0.0
description: Custom skill for specific tasks
entry_point: skill:MySkillPlugin
agent_role: coder
triggers:
  - "create widget"
  - "build component"
```

### Implementation
```python
# skill.py
from hafs.plugins import PluginBase

class MySkillPlugin(PluginBase):
    async def execute(self, context, prompt):
        # Skill implementation
        return result
```

## 6. Skill Access Paths

| Resource | Path |
|----------|------|
| Claude Skills Config | `~/.claude/settings.json > enabledPlugins` |
| Claude Plans | `~/.claude/plans/` |
| hafs Plugins | `~/Code/hafs/plugins/` |
| Plugin Config | `~/.config/hafs/config.toml > [plugins]` |

## 7. Cross-Agent Skill Sharing

Skills invoked by Claude can share results with other agents via AFS:

1. Skill outputs written to `scratchpad/state.md`
2. Persistent results stored in `hivemind/knowledge.json`
3. Other agents (Gemini, GPT) read from AFS to access skill results

## 8. Best Practices

1. **Check skill availability** before attempting invocation
2. **Use AFS for skill results** - write outputs to scratchpad for cross-agent access
3. **Map skills to roles** - assign appropriate hAFS roles for skill-capable agents
4. **Fallback gracefully** - if skill unavailable, use standard agent capabilities

## 9. Resources

### Official Documentation
- [Claude Code Plugins](https://www.anthropic.com/news/claude-code-plugins)
- [Skills Explained](https://claude.com/blog/skills-explained)
- [MCP Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp)

### Plugin Marketplaces
- [Claude Plugins Dev](https://claude-plugins.dev/) - Browse & install
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)

### Awesome Lists
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) - 7260+ servers
- [claude-code-plugins-plus](https://github.com/jeremylongshore/claude-code-plugins-plus) - 243+ plugins

### MCP Server Directories
- [MCP.so](https://mcp.so) - Searchable directory
- [PulseMCP](https://www.pulsemcp.com) - Curated servers
- [Glama MCP](https://glama.ai/mcp/servers) - By category
