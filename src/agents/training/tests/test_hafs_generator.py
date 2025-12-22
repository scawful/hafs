"""Tests for the HafsSystemGenerator."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from agents.training.generators.hafs_generator import HafsSystemGenerator
from config.schema import HafsConfig, GeneralConfig, ProjectConfig

@pytest.mark.asyncio
async def test_hafs_generator_extracts_from_config():
    """Verify generator extracts items from config."""
    
    # Mock config
    mock_config = HafsConfig(
        general=GeneralConfig(
            cli_aliases={
                "hc": "hafs-cli chat",
                "htw": "hafs-cli training status --watch"
            }
        ),
        projects=[
            ProjectConfig(name="TestProject", path=Path("/tmp/test_project"))
        ]
    )

    with patch("agents.training.generators.hafs_generator.load_config", return_value=mock_config):
        generator = HafsSystemGenerator()
        # Mock setup to avoid orchestrator init
        generator.setup = AsyncMock() 
        
        items = await generator.extract_source_items()
        
        # Check aliases
        aliases = [i for i in items if i.category == "alias"]
        assert len(aliases) == 2
        assert any(i.name == "hc" and i.command == "hafs-cli chat" for i in aliases)
        
        # Check standard CLI commands (should be present as fallback/standard)
        cli_cmds = [i for i in items if i.category == "cli"]
        assert len(cli_cmds) >= 5 # "hafs-cli chat" etc.
        
        # Check projects
        projects = [i for i in items if i.category == "project"]
        assert len(projects) == 1
        assert projects[0].name == "TestProject"
        assert projects[0].command == "cd /tmp/test_project"

@pytest.mark.asyncio
async def test_hafs_generator_build_script_detection():
    """Verify build script items are generated if file exists."""
    
    mock_config = HafsConfig(
        projects=[
            ProjectConfig(name="BuildProject", path=Path("/tmp/build_project"))
        ]
    )

    with patch("agents.training.generators.hafs_generator.load_config", return_value=mock_config):
        with patch("pathlib.Path.exists", return_value=True): # Mock file existence
            generator = HafsSystemGenerator()
            generator.setup = AsyncMock()
            
            items = await generator.extract_source_items()
            
            build_items = [i for i in items if i.category == "project-build"]
            assert len(build_items) == 1
            assert build_items[0].name == "Build BuildProject"
            assert "./build.sh" in build_items[0].command
