import unittest
from unittest.mock import MagicMock, patch
import sys
import os

from core.registry import AgentRegistry
from agents.core.base import BaseAgent

class MockAgent(BaseAgent):
    def __init__(self):
        super().__init__("MockAgent", "Test agent")
    async def run_task(self): return "Success"

class TestCoreArchitecture(unittest.TestCase):
    
    def setUp(self):
        self.registry = AgentRegistry()

    def test_registry_registration(self):
        """Test that agents can be registered and retrieved."""
        self.registry.register_agent(MockAgent)
        
        agents = self.registry.list_agents()
        self.assertIn("MockAgent", agents)
        self.assertEqual(agents["MockAgent"], MockAgent)
        
        instance = self.registry.get_agent("MockAgent")
        self.assertIsInstance(instance, MockAgent)

    def test_registry_override(self):
        """Test that registering with the same name overrides the previous entry."""
        class MockAgent2(BaseAgent):
            pass
            
        self.registry.register_agent(MockAgent)
        self.registry.register_agent(MockAgent2, name_override="MockAgent")
        
        agents = self.registry.list_agents()
        self.assertEqual(agents["MockAgent"], MockAgent2)

    def test_registry_missing(self):
        """Test handling of missing agents."""
        with self.assertRaises(ValueError):
            self.registry.get_agent("NonExistentAgent")

    def test_adapter_registration(self):
        """Test adapter registration."""
        class MockAdapter: pass
        self.registry.register_adapter("test_adapter", MockAdapter)
        
        retrieved = self.registry.get_adapter("test_adapter")
        self.assertIsInstance(retrieved, MockAdapter)

if __name__ == '__main__':
    unittest.main()
