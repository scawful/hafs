"""Agent and Plugin Registry for HAFS."""
from typing import Any, Dict, Type

from hafs.adapters.base import AdapterRegistry
from hafs.agents.base import BaseAgent

# Import context building agents for registration
from hafs.agents.context_report_pipeline import ContextReportPipeline
from hafs.agents.alttp_module_analyzer import ALTTPModuleAnalyzer
from hafs.agents.oracle_kb_builder import OracleKBBuilder
from hafs.agents.oracle_analyzer import OracleOfSecretsAnalyzer
from hafs.agents.report_manager import ReportManager

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
        try:
            AdapterRegistry.register(adapter_class)
        except Exception:
            pass

    def get_agent(self, name: str) -> BaseAgent:
        """Instantiates and returns an agent by name."""
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self.agents[name]()

    def get_adapter(self, name: str) -> Any:
        """Instantiates and returns an adapter by name."""
        if name in self.adapters:
            return self.adapters[name]()
        adapter = AdapterRegistry.get(name)
        if adapter:
            return adapter
        raise ValueError(f"Adapter '{name}' not found in registry.")

    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        """Returns the dictionary of registered agent classes."""
        return self.agents

# Global instance to be used by plugins
agent_registry = AgentRegistry()

# Register context building agents
agent_registry.register_agent(ContextReportPipeline)
agent_registry.register_agent(ALTTPModuleAnalyzer, "alttp_module_analyzer")
agent_registry.register_agent(OracleKBBuilder, "oracle_kb_builder")
agent_registry.register_agent(OracleOfSecretsAnalyzer, "oracle_analyzer")
agent_registry.register_agent(ReportManager, "report_manager")
