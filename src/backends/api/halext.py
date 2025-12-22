from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Optional

from backends.api.openai import OpenAIBackend

logger = logging.getLogger(__name__)


class HalextBackend(OpenAIBackend):
    """Backend for halext-org AI gateway.
    
    Inherits from OpenAIBackend as the gateway is OpenAI-compatible
    but adds specific authentication and routing headers.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        model: str = "default",
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 60,
    ):
        # Use halext-org default if no base_url provided
        resolved_url = base_url or "https://api.halext.org/v1"
        
        super().__init__(
            base_url=resolved_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        self.org_id = org_id

    @property
    def name(self) -> str:
        return "halext"

    def _get_headers(self) -> dict[str, str]:
        """Get headers for the request, including org_id."""
        headers = super()._get_headers()
        if self.org_id:
            headers["X-Halext-Org-ID"] = self.org_id
        # Add any other required gateway headers here
        headers["X-Halext-Client"] = "hafs-agent"
        return headers
