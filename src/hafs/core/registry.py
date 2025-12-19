"""Agent and Plugin Registry for HAFS."""
from typing import Dict, Any, Type
from hafs.agents.base import BaseAgent

class AgentRegistry:
    """A registry for discovering and managing agents."""

    def __init__(self):
        self.agents: Dict[str, Type[BaseAgent]] = {}
        self.adapters: Dict[str, Any] = {}

    def register_agent(self, agent_class: Type[BaseAgent], name_override: str = None):
        """Registers a new agent class."""
        name = name_override or agent_class.__name__
        self.agents[name] = agent_class

    def register_adapter(self, name: str, adapter_class: Any):
        """Registers a new adapter class."""
        self.adapters[name] = adapter_class

    def get_agent(self, name: str) -> BaseAgent:
        """Instantiates and returns an agent by name."""
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self.agents[name]()

    def get_adapter(self, name: str) -> Any:
        """Instantiates and returns an adapter by name."""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found in registry.")
        return self.adapters[name]()

    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        """Returns the dictionary of registered agent classes."""
        return self.agents

# Global instance to be used by plugins
agent_registry = AgentRegistry()