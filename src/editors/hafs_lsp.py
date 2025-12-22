#!/usr/bin/env python3
"""
hafs-lsp: Custom Language Server for 65816 ASM and ALTTP development

Features:
- Code completion with ROM context from yaze/mesen2
- LLVM-based ASM parsing and analysis
- Integration with hafs knowledge graph
- Multi-model support (1.5B for speed, 7B for quality)
- Terminal integration for shell completions
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    CompletionItem,
    CompletionList,
    CompletionParams,
    Position,
    Range,
)
from pygls.server import LanguageServer

# hafs imports
from services.local_ai_orchestrator import (
    LocalAIOrchestrator,
    InferenceRequest,
    RequestPriority,
)

logger = logging.getLogger(__name__)


class HafsLanguageServer(LanguageServer):
    """Custom LSP server with ROM-aware completions."""

    def __init__(self):
        super().__init__("hafs-lsp", "v0.1.0")

        # Load configuration
        from editors.config import get_config
        self.config = get_config()

        # Check if enabled
        if not self.config.enabled:
            logger.warning("hafs-lsp is DISABLED in config/lsp.toml")
            logger.warning("Set server.enabled = true to activate")
            return

        # AI orchestrator for completions
        self.orchestrator: Optional[LocalAIOrchestrator] = None

        # Context providers (MCP servers)
        self.yaze_context: Optional[Any] = None
        self.mesen_context: Optional[Any] = None
        self.kg_context: Optional[Any] = None

        logger.info("hafs-lsp initialized (config loaded from lsp.toml)")

    async def setup(self):
        """Initialize the LSP server."""
        if not self.config.enabled:
            logger.error("Cannot start: hafs-lsp is disabled in config")
            return

        # Get default model from config
        default_model = self.config.get_model(context_length=0, manual=False)

        # Start AI orchestrator
        self.orchestrator = LocalAIOrchestrator(
            ollama_url="http://localhost:11434",
            default_model=default_model or "qwen2.5-coder:1.5b",
        )
        await self.orchestrator.start()

        # Connect to MCP servers if ROM context enabled
        if self.config.get_context_settings().get("enable_rom_context"):
            await self._connect_mcp_servers()

        logger.info(f"hafs-lsp ready (strategy: {self.config.strategy})")

    async def _connect_mcp_servers(self):
        """Connect to yaze, mesen2, and book-of-mudora MCP servers."""
        # TODO: Use MCP client to connect to:
        # - yaze-debugger (ROM state, breakpoints, memory)
        # - mesen2 (emulator state, PPU viewer)
        # - book-of-mudora (knowledge graph, symbols)
        pass

    async def get_completion(
        self,
        document_uri: str,
        position: Position,
        context: str,
    ) -> CompletionList:
        """Generate code completion."""

        # Determine if this is ASM or general code
        is_asm = document_uri.endswith(('.asm', '.s'))

        # Get ROM context if available
        rom_context = await self._get_rom_context(position) if is_asm else None

        # Build FIM prompt
        prompt = self._build_fim_prompt(context, position, rom_context)

        # Choose model based on complexity
        model = self._select_model(context, is_asm)

        # Submit inference request
        request = InferenceRequest(
            id=f"completion_{position.line}_{position.character}",
            priority=RequestPriority.INTERACTIVE,  # High priority for autocomplete
            prompt=prompt,
            model=model,
            max_tokens=128,  # Short completions
            temperature=0.2,  # Low temp for consistency
        )

        result = await self.orchestrator.submit_request(request)

        if result.error:
            logger.error(f"Completion failed: {result.error}")
            return CompletionList(is_incomplete=False, items=[])

        # Parse completion into items
        items = self._parse_completion(result.response)

        return CompletionList(is_incomplete=False, items=items)

    async def _get_rom_context(self, position: Position) -> Optional[Dict[str, Any]]:
        """Get current ROM state from yaze/mesen2."""
        if not self.yaze_context:
            return None

        # Query yaze for:
        # - Current bank
        # - Valid memory addresses at this location
        # - Nearby symbols/routines
        # - Register states if in debugger

        return {
            "bank": 0x00,  # TODO: Get from yaze
            "pc": 0x8000,  # TODO: Get from debugger
            "symbols": [],  # TODO: Get from knowledge graph
        }

    def _build_fim_prompt(
        self,
        context: str,
        position: Position,
        rom_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build fill-in-the-middle prompt."""

        # Split context at cursor position
        lines = context.split('\n')
        prefix = '\n'.join(lines[:position.line])
        suffix = '\n'.join(lines[position.line + 1:])

        # Add ROM context if available
        context_str = ""
        if rom_context:
            context_str = f"\n; Current bank: ${rom_context['bank']:02X}\n"
            context_str += f"; PC: ${rom_context['pc']:04X}\n"

        # FIM format for Qwen2.5-Coder
        prompt = f"<fim_prefix>{context_str}{prefix}<fim_suffix>{suffix}<fim_middle>"

        return prompt

    def _select_model(self, context: str, is_asm: bool, manual: bool = False) -> str:
        """Choose model based on config strategy."""

        # Get model from config
        model = self.config.get_model(
            context_length=len(context),
            manual=manual
        )

        # If config says no auto-complete and not manual, return None
        if model is None:
            return None

        # Check for custom fine-tuned model
        if len(context) < 200:
            custom = self.config.get_custom_model("fast")
            if custom:
                logger.info(f"Using custom fast model: {custom}")
                return custom

        custom = self.config.get_custom_model("quality")
        if custom:
            logger.info(f"Using custom quality model: {custom}")
            return custom

        return model

    def _parse_completion(self, response: str) -> List[CompletionItem]:
        """Parse model output into completion items."""

        # Extract completion from FIM response
        if '<fim_middle>' in response:
            completion = response.split('<fim_middle>')[-1]
            completion = completion.split('<|endoftext|>')[0]
        else:
            completion = response

        # Create completion item
        items = [
            CompletionItem(
                label=completion.strip(),
                insert_text=completion.strip(),
                detail="hafs-lsp (AI-powered)",
            )
        ]

        return items


# LSP server setup
server = HafsLanguageServer()


@server.feature(TEXT_DOCUMENT_COMPLETION)
async def completions(params: CompletionParams) -> CompletionList:
    """Handle completion requests."""
    document = server.workspace.get_document(params.text_document.uri)

    # Get context around cursor
    context = document.source

    return await server.get_completion(
        document.uri,
        params.position,
        context,
    )


async def main():
    """Start the LSP server."""
    await server.setup()
    server.start_io()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
