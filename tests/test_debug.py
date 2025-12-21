
import sys
import os

print("Hello from test_debug.py")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

try:
    from agents.core.base import BaseAgent
    print("BaseAgent imported successfully")
except ImportError as e:
    print(f"Failed to import BaseAgent: {e}")

try:
    from agents.knowledge.alttp_unified import UnifiedALTTPKnowledge
    print("UnifiedALTTPKnowledge imported successfully")
except ImportError as e:
    print(f"Failed to import UnifiedALTTPKnowledge: {e}")
