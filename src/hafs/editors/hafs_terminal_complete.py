#!/usr/bin/env python3
"""
Terminal completion for hafs-lsp

Provides shell completions for hex addresses and ASM mnemonics.

Usage in ~/.zshrc:
  function _hafs_complete() {
    local response=$(echo "$BUFFER" | python3 ~/Code/hafs/src/hafs/editors/hafs_terminal_complete.py)
    BUFFER="${BUFFER}${response}"
    CURSOR=$#BUFFER
  }
  bindkey '^ ' _hafs_complete  # Ctrl+Space
"""

import sys
import asyncio
from pathlib import Path

# Add hafs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hafs.services.local_ai_orchestrator import (
    LocalAIOrchestrator,
    InferenceRequest,
    RequestPriority,
)


async def get_completion(context: str) -> str:
    """Get completion from LSP model."""

    orchestrator = LocalAIOrchestrator(
        ollama_url="http://localhost:11434",
        default_model="qwen2.5-coder:1.5b",  # Fast model for terminal
    )

    await orchestrator.start()

    # Build simple prompt for terminal completion
    prompt = f"Complete this 65816 assembly instruction: {context}"

    request = InferenceRequest(
        id="terminal_completion",
        priority=RequestPriority.INTERACTIVE,
        prompt=prompt,
        model="qwen2.5-coder:1.5b",
        max_tokens=32,  # Short completions only
        temperature=0.0,  # Deterministic
    )

    result = await orchestrator.submit_request(request)

    await orchestrator.stop()

    if result.error:
        return ""

    # Extract just the completion
    completion = result.response.strip().split('\n')[0]

    return completion


def main():
    """Read from stdin and return completion."""
    context = sys.stdin.read().strip()

    # Only complete if it looks like ASM
    if not any(keyword in context.upper() for keyword in ['LDA', 'STA', 'JMP', 'JSR', '$', '#']):
        return

    try:
        completion = asyncio.run(get_completion(context))
        print(completion, end='')
    except Exception:
        # Fail silently in terminal
        pass


if __name__ == "__main__":
    main()
