# HAFS Plugin & Adapter Development Guide

This guide explains how to create adapter plugins that integrate external tools and services with HAFS.

---

## 1. Architecture Overview

HAFS uses a **plugin-based adapter architecture**. Core agents use generic stub adapters that return empty data. Plugins register real implementations to enable full functionality with your specific tools.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `AgentRegistry` | `hafs.core.registry` | Central registry for agents and adapters |
| `BaseAdapter` | `hafs.adapters.base` | Abstract interface for all adapters |
| `IntegrationPlugin` | `hafs.plugins.protocol` | Typed adapter plugin interface |
| `adapter protocols` | `hafs.adapters.protocols` | Standard issue/review/search methods |
| `GenericAdapter` | Inline in agents | Placeholder stubs (replaced by plugins) |

### Registration Flow

```
Plugin Load → IntegrationPlugin.get_* → registry.register_adapter("name", AdapterClass)
             (or legacy register(registry)) → registry.register_agent(AgentClass)
```

---

## 2. Standard Adapter Names

HAFS agents look up adapters by these standard names:

| Adapter Name | Purpose | Used By |
|--------------|---------|---------|
| `issue_tracker` | Bug/issue tracking system | `DailyBriefingAgent`, `TrendWatcher` |
| `code_review` | Code review system | `DailyBriefingAgent`, `TrendWatcher`, `AutonomousContextAgent` |
| `code_search` | Codebase search/indexing | `TrendWatcher`, `AutonomousContextAgent` |

### Registry Lookup Pattern

```python
# Agents look up adapters dynamically
async def setup(self):
    try:
        self.issue_tracker = agent_registry.get_adapter("issue_tracker")
        await self.issue_tracker.connect()
    except:
        pass  # Graceful fallback if adapter not registered
```

---

## 3. Adapter Interface Specifications

### 3.1 Issue Tracker Adapter

Integrates with bug/issue tracking systems (Jira, GitHub Issues, Linear, etc.).

Legacy adapters that only implement `search_bugs(query)` are still supported via
`hafs.adapters.helpers.search_issues`.

```python
from hafs.adapters.protocols import IssueRecord

class IssueTrackerAdapter:
    """Adapter for issue tracking systems."""

    @property
    def name(self) -> str:
        return "issue_tracker"

    async def connect(self) -> bool:
        """Establish connection to issue tracker."""
        pass

    async def search_issues(self, query: str, limit: int = 50) -> list[IssueRecord]:
        """Search for issues matching query.

        Args:
            query: Search query string (e.g., "assignee:me status:open")
            limit: Max results
        """
        pass

    async def disconnect(self) -> None:
        """Clean up connection."""
        pass
```

### 3.2 Code Review Adapter

Integrates with code review systems (GitHub PRs, GitLab MRs, Gerrit, etc.).

```python
class CodeReviewAdapter:
    """Adapter for code review systems."""

    @property
    def name(self) -> str:
        return "code_review"

    async def connect(self) -> bool:
        pass

    async def get_submitted(self, user: str, limit: int = 5) -> list[dict]:
        """Get recently submitted reviews by user.

        Args:
            user: Username/identifier
            limit: Maximum results to return

        Returns:
            List of review records (typically last 24h)
        """
        pass

    async def get_reviews(self, user: str | None = None) -> list[dict]:
        """Get pending reviews for user.

        Args:
            user: Optional username filter

        Returns:
            List of review records awaiting review
        """
        pass

    async def disconnect(self) -> None:
        pass
```

### 3.3 Code Search Adapter

Integrates with code search/indexing systems (Sourcegraph, OpenGrok, ripgrep server, etc.).

```python
class CodeSearchAdapter:
    """Adapter for code search systems."""

    @property
    def name(self) -> str:
        return "code_search"

    async def connect(self) -> bool:
        pass

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search codebase for query.

        Args:
            query: Search term (symbol, text, or file pattern)
            limit: Maximum results
        """
        pass

    async def read_file(self, path: str) -> str:
        """Read file content.

        Args:
            path: Path to file in repository

        Returns:
            File content as string
        """
        pass

    async def disconnect(self) -> None:
        pass
```

---

## 4. Plugin Structure

### 4.1 Directory Layout

```
my_hafs_plugin/
├── __init__.py
├── hafs_plugin.py          # Legacy entry point (optional)
├── plugin.py               # Plugin class entry (recommended)
├── adapters/
│   ├── __init__.py
│   ├── issues.py           # IssueTrackerAdapter impl
│   ├── reviews.py          # CodeReviewAdapter impl
│   └── search.py           # CodeSearchAdapter impl
├── agents/
│   ├── __init__.py
│   └── custom_agent.py     # Custom agents
└── config.py               # Plugin-specific config
```

### 4.2 Plugin Entry Point

Plugins can expose either:
- A `Plugin` class implementing `HafsPlugin` + `IntegrationPlugin` (recommended)
- A legacy `register(registry)` function (still supported)

**Recommended (Plugin class):**

```python
# my_hafs_plugin/plugin.py
from hafs.plugins.protocol import HafsPlugin, IntegrationPlugin

class Plugin(HafsPlugin, IntegrationPlugin):
    @property
    def name(self) -> str:
        return "my_hafs_plugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    def activate(self, app) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def get_issue_tracker(self):
        from .adapters.issues import MyIssueTracker
        return MyIssueTracker

    def get_code_review(self):
        from .adapters.reviews import MyCodeReview
        return MyCodeReview

    def get_code_search(self):
        from .adapters.search import MyCodeSearch
        return MyCodeSearch
```

**Legacy (register function):**

```python
# my_hafs_plugin/hafs_plugin.py

from .adapters.issues import MyIssueTracker
from .adapters.reviews import MyCodeReview
from .adapters.search import MyCodeSearch
from .agents.custom_agent import MyCustomAgent

def register(registry):
    """Called by HAFS plugin loader.

    Args:
        registry: The global agent_registry instance
    """
    # Register adapters with standard names
    registry.register_adapter("issue_tracker", MyIssueTracker)
    registry.register_adapter("code_review", MyCodeReview)
    registry.register_adapter("code_search", MyCodeSearch)

    # Register custom agents
    registry.register_agent(MyCustomAgent)
```

### 4.3 Plugin Configuration

Plugins can read custom config sections from `~/.config/hafs/config.toml`
(preferred) or project-local `hafs.toml`:

```toml
# User's config.toml

[plugins]
enabled_plugins = ["my_hafs_plugin"]

[my_plugin]
api_url = "https://api.example.com"
api_key_env = "MY_API_KEY"
default_project = "my-project"
```

```python
# In your plugin
from pathlib import Path
import tomllib

def get_plugin_config():
    path = Path.home() / ".config" / "hafs" / "config.toml"
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text())
    return data.get("my_plugin", {})

api_url = get_plugin_config().get("api_url")
```

---

## 5. Extending Core Agents

### 5.1 Override Pattern

Create extended agents that inject your adapters:

```python
from hafs.agents.context_builder import AutonomousContextAgent
from hafs.core.registry import agent_registry

class EnhancedContextAgent(AutonomousContextAgent):
    """Extended agent with real adapter connections."""

    # Custom system context for your team
    SYSTEM_CONTEXT = """
    You are an autonomous background agent.
    Your job is to explore the work environment and maintain context documentation.

    # TEAM CONTEXT
    - Primary workspace: {workspace_path}
    - Key projects: {projects}

    # INSTRUCTIONS
    - Focus on active work items and their dependencies
    - Track code changes and their impact
    """

    async def setup(self):
        await super().setup()

        # Replace stubs with real adapters
        self.bugs = agent_registry.get_adapter("issue_tracker")
        self.critique = agent_registry.get_adapter("code_review")
        self.codesearch = agent_registry.get_adapter("code_search")

        if self.bugs:
            await self.bugs.connect()
        if self.critique:
            await self.critique.connect()
        if self.codesearch:
            await self.codesearch.connect()

    async def gather_inventory(self):
        """Override with enhanced data gathering."""
        result = await super().gather_inventory()

        # Add custom critical file checks
        if self.codesearch:
            critical_files = self._get_critical_files()
            for fpath in critical_files:
                try:
                    content = await self.codesearch.read_file(fpath)
                    self.doc_context[fpath] = content[:5000]
                except:
                    pass

        return result

    def _get_critical_files(self):
        """Return list of critical files to always monitor."""
        return [
            "README.md",
            "docs/architecture/ARCHITECTURE.md",
            # Add your team's critical files
        ]
```

### 5.2 Shadow Observer Extension

Extend to detect your specific tools:

```python
from hafs.agents.shadow_observer import ShadowObserver

class EnhancedShadowObserver(ShadowObserver):
    """Observer with custom command detection."""

    async def process_command(self, raw_line: str):
        await super().process_command(raw_line)

        cmd = raw_line.split(";", 1)[1].strip() if ";" in raw_line else raw_line

        # Detect navigation to key directories
        if cmd.startswith("cd ") and "/my-workspace/" in cmd:
            print(f"[{self.name}] Detected workspace navigation")
            # Update context

        # Detect build commands
        if "make build" in cmd or "npm run build" in cmd:
            target = self._extract_target(cmd)
            print(f"[{self.name}] Detected build: {target}")
            # Log build context

        # Detect test runs
        if "pytest" in cmd or "npm test" in cmd:
            print(f"[{self.name}] Detected test run")
            # Log test context
```

---

## 6. Example: GitHub Adapter Plugin

Complete example of a GitHub-based plugin:

```python
# github_hafs_plugin/adapters/github_issues.py

import aiohttp
from dataclasses import dataclass
from typing import List
import os

@dataclass
class GitHubIssue:
    id: str
    title: str
    priority: str
    status: str

class GitHubIssueAdapter:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("GITHUB_REPO", "owner/repo")
        self.session = None

    @property
    def name(self) -> str:
        return "issue_tracker"

    async def connect(self) -> bool:
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"token {self.token}"}
        )
        return True

    async def search_issues(self, query: str, limit: int = 50) -> List[GitHubIssue]:
        # Parse query (e.g., "assignee:me status:open")
        params = self._parse_query(query)

        url = f"https://api.github.com/repos/{self.repo}/issues"
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()

        return [
            GitHubIssue(
                id=str(issue["number"]),
                title=issue["title"],
                priority=self._get_priority(issue),
                status=issue["state"]
            )
            for issue in data[:limit]
        ]

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()

    def _parse_query(self, query: str) -> dict:
        params = {"state": "open"}
        if "assignee:me" in query:
            params["assignee"] = "@me"
        return params

    def _get_priority(self, issue: dict) -> str:
        labels = [l["name"] for l in issue.get("labels", [])]
        if "priority:critical" in labels:
            return "P0"
        elif "priority:high" in labels:
            return "P1"
        return "P2"
```

```python
# github_hafs_plugin/hafs_plugin.py

from .adapters.github_issues import GitHubIssueAdapter
from .adapters.github_prs import GitHubPRAdapter

def register(registry):
    registry.register_adapter("issue_tracker", GitHubIssueAdapter)
    registry.register_adapter("code_review", GitHubPRAdapter)
```

---

## 7. Testing Your Plugin

### 7.1 Verify Registration

```python
from hafs.core.registry import agent_registry
from hafs.core.plugin_loader import load_plugins

load_plugins()

# Check adapters
print("Registered adapters:", list(agent_registry.adapters.keys()))
# Expected: ['issue_tracker', 'code_review', 'code_search']

# Check agents
print("Registered agents:", list(agent_registry.list_agents().keys()))
```

### 7.2 Test Connections

```python
import asyncio

async def test_adapters():
    tracker = agent_registry.get_adapter("issue_tracker")
    if tracker:
        await tracker.connect()
        issues = await tracker.search_issues("assignee:me status:open")
        print(f"Found {len(issues)} issues")
        await tracker.disconnect()

asyncio.run(test_adapters())
```

### 7.3 Integration Test

```python
from hafs.agents.daily_briefing import DailyBriefingAgent
import asyncio

async def test_briefing():
    agent = DailyBriefingAgent()
    await agent.setup()
    result = await agent.run_task()
    print(result)

asyncio.run(test_briefing())
```

---

## 8. Best Practices

1. **Use standard adapter names** - Stick to `issue_tracker`, `code_review`, `code_search` for compatibility
2. **Handle missing adapters gracefully** - Agents should work (with limited functionality) without adapters
3. **Don't hardcode credentials** - Use environment variables or config file references
4. **Implement `disconnect()`** - Clean up connections properly
5. **Log adapter activity** - Use HAFS logging for debugging
6. **Test independently** - Test adapters before integrating with agents

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create plugin package with a `Plugin` class (or legacy `hafs_plugin.py`) |
| 2 | Implement adapter classes for your tools |
| 3 | Register adapters with standard names |
| 4 | Add plugin to `[plugins].enabled_plugins` in config |
| 5 | Test adapter connections |
| 6 | Optionally extend core agents for enhanced functionality |
