"""Base class for background agents."""

from __future__ import annotations

import json
import logging
import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a background agent."""

    enabled: bool
    provider: str
    model: str
    schedule: str
    description: str
    tasks: dict[str, Any]
    capabilities: dict[str, bool]


class BackgroundAgent(ABC):
    """Base class for all background agents.

    Background agents run autonomously to:
    - Explore and catalog codebases
    - Build and update knowledge bases
    - Monitor system health
    - Sync state between machines
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize background agent.

        Args:
            config_path: Path to agent configuration TOML file
            verbose: Enable verbose logging
        """
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        self.verbose = verbose
        self.config = self._load_config()
        self.agent_name = self.__class__.__name__.replace("Agent", "").lower()

        # Setup logging
        self._setup_logging()

        # Ensure output directories exist
        self._ensure_directories()

    def _default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path("config/windows_background_agents.toml")

    def _load_config(self) -> AgentConfig:
        """Load agent configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)

        agent_key = f"agents.{self.agent_name}"
        if agent_key not in data.get("agents", {}):
            raise ValueError(f"No configuration found for agent: {self.agent_name}")

        agent_data = data["agents"][self.agent_name]
        return AgentConfig(
            enabled=agent_data.get("enabled", False),
            provider=agent_data.get("provider", "claude"),
            model=agent_data.get("model", "claude-sonnet-4-5"),
            schedule=agent_data.get("schedule", "0 */6 * * *"),
            description=agent_data.get("description", ""),
            tasks=agent_data.get("tasks", {}),
            capabilities=agent_data.get("capabilities", {}),
        )

    def _setup_logging(self):
        """Setup logging for this agent."""
        log_level = logging.DEBUG if self.verbose else logging.INFO

        # Create log directory
        log_dir = Path(self.config.tasks.get("report_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.agent_name}_{timestamp}.log"

        # Configure logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger.info(f"Starting {self.agent_name} agent")
        logger.info(f"Configuration: {self.config.description}")
        logger.info(f"Provider: {self.config.provider} ({self.config.model})")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        for key in ["output_dir", "report_dir"]:
            if key in self.config.tasks:
                path = Path(self.config.tasks[key])
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")

    def _get_api_key(self) -> str:
        """Get API key for configured provider."""
        if self.config.provider == "claude":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return key
        elif self.config.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set")
            return key
        elif self.config.provider == "gemini":
            # Should not be used for background agents!
            logger.warning("Gemini should be reserved for training data generation")
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY not set")
            return key
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _save_output(self, data: dict[str, Any], filename: str):
        """Save agent output to JSON file.

        Args:
            data: Data to save
            filename: Output filename (without extension)
        """
        output_dir = Path(self.config.tasks.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{filename}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved output to: {output_file}")
        return output_file

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the agent's main task.

        Returns:
            Dictionary with execution results and metrics
        """
        pass

    def execute(self) -> dict[str, Any]:
        """Execute the agent with error handling.

        Returns:
            Dictionary with execution results and status
        """
        if not self.config.enabled:
            logger.warning(f"{self.agent_name} agent is disabled in configuration")
            return {"status": "disabled", "message": "Agent is disabled in config"}

        try:
            logger.info(f"Executing {self.agent_name} agent")
            result = self.run()
            logger.info(f"{self.agent_name} agent completed successfully")
            return {
                "status": "success",
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
        except Exception as e:
            logger.error(f"{self.agent_name} agent failed: {e}", exc_info=True)
            return {
                "status": "error",
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
