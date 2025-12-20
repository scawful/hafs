"""Pydantic configuration models for HAFS."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PolicyType(str, Enum):
    """AFS directory permission policy."""

    READ_ONLY = "read_only"
    WRITABLE = "writable"
    EXECUTABLE = "executable"


# ============================================================================
# Orchestration Configuration Models
# ============================================================================


class BackendConfig(BaseModel):
    """Configuration for a chat backend."""

    name: str
    enabled: bool = True
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    working_dir: Optional[Path] = None


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    role: Literal["general", "planner", "coder", "critic", "researcher"] = "general"
    backend: str = "gemini"
    system_prompt: str = ""
    auto_start: bool = False


class OrchestratorConfig(BaseModel):
    """Configuration for multi-agent orchestration."""

    enabled: bool = True
    max_agents: int = 5
    default_agents: list[AgentConfig] = Field(default_factory=list)
    transactive_memory_size: int = 50
    auto_routing: bool = True


class SynergyConfig(BaseModel):
    """Configuration for synergy/ToM subsystem."""

    enabled: bool = True
    profile_storage: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "hafs" / "profiles"
    )
    marker_confidence_threshold: float = 0.7
    score_display: bool = True


class PluginConfig(BaseModel):
    """Configuration for plugins."""

    enabled_plugins: list[str] = Field(default_factory=list)
    plugin_dirs: list[Path] = Field(default_factory=list)


class ServiceConfig(BaseModel):
    """Configuration for a managed service."""

    name: str
    enabled: bool = True
    auto_start: bool = False  # Start at login
    environment: dict[str, str] = Field(default_factory=dict)


class ServicesConfig(BaseModel):
    """Configuration for service management."""

    enabled: bool = True
    services: dict[str, ServiceConfig] = Field(
        default_factory=lambda: {
            "orchestrator": ServiceConfig(name="orchestrator"),
            "coordinator": ServiceConfig(name="coordinator"),
            "autonomy": ServiceConfig(name="autonomy"),
            "dashboard": ServiceConfig(name="dashboard"),
        }
    )


class AFSDirectoryConfig(BaseModel):
    """Configuration for a single AFS directory type."""

    name: str
    policy: PolicyType
    description: str = ""


class ParserConfig(BaseModel):
    """Configuration for a log parser."""

    enabled: bool = True
    base_path: Optional[Path] = None
    max_items: int = 50


class ThemeConfig(BaseModel):
    """UI theme configuration."""

    primary: str = "#4C3B52"
    secondary: str = "#9B59B6"
    accent: str = "#E74C3C"
    gradient_start: str = "#4C3B52"
    gradient_end: str = "#000000"


class WorkspaceDirectory(BaseModel):
    """A directory in the workspace for file browsing."""

    path: Path
    name: Optional[str] = None  # Display name, defaults to folder name
    recursive: bool = True  # Whether to show subdirectories


class ToolProfileConfig(BaseModel):
    """Tool access profile for background agents."""

    name: str
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    description: str = ""


class SkillConfig(BaseModel):
    """Reusable skill definition for personas."""

    name: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class ExecutionModeConfig(BaseModel):
    """Execution mode maps to a tool profile."""

    name: str
    tool_profile: str
    description: str = ""


class PersonaConfig(BaseModel):
    """Persona definition for agent roles and prompts."""

    name: str
    role: Literal["general", "planner", "coder", "critic", "researcher"] = "general"
    description: str = ""
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)
    tool_profile: Optional[str] = None
    execution_mode: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    default_for_role: bool = False
    enabled: bool = True


class ProjectConfig(BaseModel):
    """Configuration for a tracked project."""

    name: str
    path: Path
    kind: str = "general"
    tags: list[str] = Field(default_factory=list)
    tooling_profile: Optional[str] = None
    knowledge_roots: list[Path] = Field(default_factory=list)
    enabled: bool = True
    description: str = ""


class GeneralConfig(BaseModel):
    """General application settings."""

    refresh_interval: int = 5
    show_hidden_files: bool = False
    default_editor: str = "nvim"
    vim_navigation_enabled: bool = False
    context_root: Path = Field(default_factory=lambda: Path.home() / ".context")
    agent_workspaces_dir: Path = Field(default_factory=lambda: Path.home() / "AgentWorkspaces")
    workspace_directories: list[WorkspaceDirectory] = Field(
        default_factory=lambda: [
            WorkspaceDirectory(path=Path.home() / "Code", name="Code"),
            WorkspaceDirectory(path=Path.home() / "Projects", name="Projects"),
        ]
    )


class ParsersConfig(BaseModel):
    """All parser configurations."""

    gemini: ParserConfig = Field(default_factory=ParserConfig)
    claude: ParserConfig = Field(default_factory=ParserConfig)
    antigravity: ParserConfig = Field(default_factory=ParserConfig)


class HafsConfig(BaseModel):
    """Root configuration model."""

    # Existing configuration
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    parsers: ParsersConfig = Field(default_factory=ParsersConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    tracked_projects: list[Path] = Field(default_factory=list)
    projects: list[ProjectConfig] = Field(default_factory=list)
    skills: list[SkillConfig] = Field(
        default_factory=lambda: [
            SkillConfig(
                name="planning",
                description="Break down tasks into steps and constraints.",
                goals=["Clarify scope", "Identify dependencies", "Define checkpoints"],
            ),
            SkillConfig(
                name="coding",
                description="Implement changes safely and clearly.",
                goals=["Write clear code", "Minimize risk", "Document intent"],
            ),
            SkillConfig(
                name="review",
                description="Find defects and risks in changes.",
                goals=["Spot regressions", "Check edge cases", "Verify coverage"],
            ),
            SkillConfig(
                name="research",
                description="Investigate systems and summarize findings.",
                goals=["Find sources", "Synthesize notes", "Capture links"],
            ),
            SkillConfig(
                name="ops",
                description="Operate services and infrastructure safely.",
                goals=["Check health", "Coordinate deployments", "Verify logs"],
            ),
        ]
    )
    personas: list[PersonaConfig] = Field(
        default_factory=lambda: [
            PersonaConfig(
                name="Generalist",
                role="general",
                skills=["planning", "research"],
                execution_mode="read_only",
                default_for_role=True,
            ),
            PersonaConfig(
                name="Planner",
                role="planner",
                skills=["planning"],
                execution_mode="read_only",
                default_for_role=True,
            ),
            PersonaConfig(
                name="Coder",
                role="coder",
                skills=["coding"],
                execution_mode="build_only",
                default_for_role=True,
            ),
            PersonaConfig(
                name="Critic",
                role="critic",
                skills=["review"],
                execution_mode="read_only",
                default_for_role=True,
            ),
            PersonaConfig(
                name="Researcher",
                role="researcher",
                skills=["research"],
                execution_mode="read_only",
                default_for_role=True,
            ),
        ]
    )
    tool_profiles: list[ToolProfileConfig] = Field(
        default_factory=lambda: [
            ToolProfileConfig(
                name="read_only",
                allow=[
                    "rg",
                    "rg_files",
                    "rg_todos",
                    "git_status",
                    "git_branch",
                    "git_log",
                    "git_diff",
                    "ls",
                ],
            )
            ,
            ToolProfileConfig(
                name="build_only",
                allow=[
                    "rg",
                    "rg_files",
                    "rg_todos",
                    "git_status",
                    "git_branch",
                    "git_log",
                    "git_diff",
                    "ls",
                    "pytest",
                    "npm_test",
                    "pnpm_test",
                    "cargo_test",
                    "go_test",
                    "make_test",
                    "just_test",
                    "npm_build",
                    "pnpm_build",
                    "cargo_build",
                    "go_build",
                    "make_build",
                    "just_build",
                ],
            ),
            ToolProfileConfig(
                name="infra_ops",
                allow=[
                    "rg",
                    "rg_files",
                    "rg_todos",
                    "git_status",
                    "git_branch",
                    "git_log",
                    "git_diff",
                    "ls",
                    "uname",
                    "whoami",
                    "uptime",
                    "df",
                    "du",
                    "ps",
                    "lsof",
                    "tail",
                    "journalctl",
                    "log_show",
                    "launchctl",
                    "systemctl",
                    "docker",
                    "docker_compose",
                    "kubectl",
                    "ssh",
                    "scp",
                    "rsync",
                    "curl",
                    "ping",
                ],
            ),
        ]
    )
    default_tool_profile: str = "read_only"
    execution_modes: list[ExecutionModeConfig] = Field(
        default_factory=lambda: [
            ExecutionModeConfig(name="read_only", tool_profile="read_only"),
            ExecutionModeConfig(name="build_only", tool_profile="build_only"),
            ExecutionModeConfig(name="infra_ops", tool_profile="infra_ops"),
        ]
    )
    default_execution_mode: str = "read_only"
    afs_directories: list[AFSDirectoryConfig] = Field(
        default_factory=lambda: [
            AFSDirectoryConfig(
                name="memory",
                policy=PolicyType.READ_ONLY,
                description="Long-term docs and specs",
            ),
            AFSDirectoryConfig(
                name="knowledge",
                policy=PolicyType.READ_ONLY,
                description="Reference materials",
            ),
            AFSDirectoryConfig(
                name="tools",
                policy=PolicyType.EXECUTABLE,
                description="Executable scripts",
            ),
            AFSDirectoryConfig(
                name="scratchpad",
                policy=PolicyType.WRITABLE,
                description="AI reasoning space",
            ),
            AFSDirectoryConfig(
                name="history",
                policy=PolicyType.READ_ONLY,
                description="Archived scratchpads",
            ),
        ]
    )

    # Orchestration configuration
    backends: list[BackendConfig] = Field(
        default_factory=lambda: [
            BackendConfig(name="gemini", command=["gemini"]),
            BackendConfig(name="claude", command=["claude"]),
            BackendConfig(name="gemini_oneshot", command=["gemini"]),
            BackendConfig(name="claude_oneshot", command=["claude"]),
        ]
    )
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    synergy: SynergyConfig = Field(default_factory=SynergyConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    def get_directory_config(self, name: str) -> Optional[AFSDirectoryConfig]:
        """Get configuration for a specific AFS directory."""
        for dir_config in self.afs_directories:
            if dir_config.name == name:
                return dir_config
        return None

    def get_backend_config(self, name: str) -> Optional[BackendConfig]:
        """Get configuration for a specific backend."""
        for backend in self.backends:
            if backend.name == name:
                return backend
        return None

    def get_enabled_backends(self) -> list[BackendConfig]:
        """Get all enabled backends."""
        return [b for b in self.backends if b.enabled]
