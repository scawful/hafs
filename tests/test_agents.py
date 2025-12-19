"""Unit tests for Public Agents."""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.agents.builder import CodeSurgeon, Toolsmith
from hafs.agents.chronos import ChronosAgent

async def test_surgeon():
    print("Testing CodeSurgeon...")
    surgeon = CodeSurgeon()
    
    # Mock LLM response
    mock_llm_response = "```python\nprint('Fixed')\n```"
    surgeon.generate_thought = MagicMock()
    f = asyncio.Future()
    f.set_result(mock_llm_response)
    surgeon.generate_thought.return_value = f
    
    # Mock file system
    test_file = Path("test_surgeon.py")
    test_file.write_text("print('Broken')")
    
    try:
        res = await surgeon.run_task(f"{test_file} | fix it")
        print(f"Result: {res}")
        if "Updated" in res and "print('Fixed')" in test_file.read_text():
            print("✅ CodeSurgeon patched file.")
        else:
            print(f"❌ CodeSurgeon failed. Content: {test_file.read_text()!r}")
    finally:
        if test_file.exists(): test_file.unlink()

async def test_toolsmith():
    print("Testing Toolsmith...")
    smith = Toolsmith()
    
    # Mock LLM
    mock_code = "```python\ndef main(): pass\n```"
    smith.generate_thought = MagicMock()
    f = asyncio.Future()
    f.set_result(mock_code)
    smith.generate_thought.return_value = f
    
    res = await smith.run_task("my_tool | do stuff")
    print(f"Result: {res}")
    
    tool_path = smith.tools_dir / "my_tool.py"
    if tool_path.exists() and "def main" in tool_path.read_text():
        print("✅ Toolsmith created tool.")
    else:
        print("❌ Toolsmith failed.")
        
    if tool_path.exists(): tool_path.unlink()

async def test_chronos():
    print("Testing Chronos...")
    chronos = ChronosAgent()
    
    # Mock subprocess (git log)
    with patch("asyncio.create_subprocess_shell") as mock_shell:
        mock_proc = MagicMock()
        
        # communicate must return an awaitable that yields (stdout, stderr)
        f_comm = asyncio.Future()
        f_comm.set_result((b"commit 123: fixed bug", b""))
        mock_proc.communicate.return_value = f_comm
        
        mock_proc.returncode = 0
        
        # create_subprocess_shell returns an awaitable that yields the process
        async def mock_create(*args, **kwargs):
            return mock_proc
        
        mock_shell.side_effect = mock_create
        
        # Mock LLM
        chronos.generate_thought = MagicMock()
        f_llm = asyncio.Future()
        f_llm.set_result("Summary: Fixed a bug.")
        chronos.generate_thought.return_value = f_llm
        
        res = await chronos.run_task("1 day")
        print(f"Result: {res}")
        
        if "Summary" in res:
            print("✅ Chronos summarized history.")
        else:
            print("❌ Chronos failed.")

async def main():
    await test_surgeon()
    print("-" * 20)
    await test_toolsmith()
    print("-" * 20)
    await test_chronos()

from pathlib import Path
if __name__ == "__main__":
    asyncio.run(main())
