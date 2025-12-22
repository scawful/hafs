"""Test Google GenAI Integration."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

genai = pytest.importorskip("google.genai")

from core.orchestrator import ModelOrchestrator
from agents.builder import Toolsmith

async def test_genai():
    print("--- Testing GenAI SDK Integration ---")
    
    # 1. Check SDK Import
    try:
        import google.genai
        print("✅ google.genai imported successfully.")
    except ImportError:
        print("⚠️ google.genai NOT found. Integration will be limited.")
        
    # 2. Test Orchestrator (Mocked if no key)
    api_key = os.environ.get("AISTUDIO_API_KEY", "dummy_key")
    
    with patch("google.genai.Client") as MockClient:
        # Setup Mock
        mock_instance = MockClient.return_value
        mock_instance.aio.models.generate_content = MagicMock()
        future = asyncio.Future()
        future.set_result(MagicMock(text="Mocked GenAI Response", usage_metadata=MagicMock(total_token_count=10)))
        mock_instance.aio.models.generate_content.return_value = future
        
        orch = ModelOrchestrator(api_key=api_key)
        
        # Force availability for test
        from core.orchestrator import GENAI_AVAILABLE
        if not GENAI_AVAILABLE:
            print("⚠️ Skipping active generation test (SDK missing).")
        else:
            print("Testing generation...")
            res = await orch.generate_content("Hello World")
            print(f"Result: {res}")
            if "Mocked" in res or "Hello" in res:
                print("✅ Orchestrator generation working.")
            else:
                print("❌ Unexpected response.")

    # 3. Test Agent Flow
    print("\n--- Testing Agent Flow (Toolsmith) ---")
    smith = Toolsmith()
    # Inject mock orchestrator
    smith.orchestrator = orch 
    
    # Since we mocked the return to be "Mocked GenAI Response", the regex in Toolsmith might fail
    # We should mock a valid tool response
    
    future_tool = asyncio.Future()
    future_tool.set_result(MagicMock(text="```python\nprint('Hello Tool')\n```", usage_metadata=MagicMock(total_token_count=10)))
    mock_instance.aio.models.generate_content.return_value = future_tool
    
    res = await smith.run_task("test_tool | print hello")
    print(f"Toolsmith Result: {res}")
    
    if "Tool created" in res:
        print("✅ Toolsmith flow working.")
    else:
        print("❌ Toolsmith failed to parse.")

if __name__ == "__main__":
    asyncio.run(test_genai())
