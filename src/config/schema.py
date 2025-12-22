"""Pydantic configuration models for HAFS."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from models.synergy_config import (
    IRTConfig,
    SynergyServiceConfig,
    ToMAssessmentConfig,
    DifficultyEstimationConfig,
)


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
    """Configuration for synergy/ToM subsystem.

    Includes research-based IRT ability estimation and LLM ToM assessment
    from "Quantifying Human-AI Synergy" paper.
    """

    enabled: bool = True
    profile_storage: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "hafs" / "profiles"
    )
    marker_confidence_threshold: float = 0.7
    score_display: bool = True

    # Research-based synergy enhancements
    tom_assessment: ToMAssessmentConfig = Field(default_factory=ToMAssessmentConfig)
    irt_estimation: IRTConfig = Field(default_factory=IRTConfig)
    difficulty_estimation: DifficultyEstimationConfig = Field(
        default_factory=DifficultyEstimationConfig
    )

    # Synergy service settings
    use_enhanced_service: bool = Field(
        default=True,
        description="Use IRT+ToM enhanced synergy service"
    )
    synergy_data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".context" / "synergy",
        description="Directory for synergy service data"
    )
    auto_start_service: bool = Field(
        default=False,
        description="Auto-start synergy service with coordinator"
    )

    @property
    def is_enhanced_enabled(self) -> bool:
        """Check if enhanced synergy tracking is fully enabled."""
        return (
            self.enabled
            and self.use_enhanced_service
            and self.tom_assessment.enabled
            and self.irt_estimation.enabled
        )


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
            "autonomy-daemon": ServiceConfig(name="autonomy-daemon"),
            "embedding-daemon": ServiceConfig(name="embedding-daemon"),
            "context-agent-daemon": ServiceConfig(name="context-agent-daemon"),
            "observability-daemon": ServiceConfig(name="observability-daemon"),
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


class ThemeVariant(str, Enum):
    """Light or dark theme variant."""

    LIGHT = "light"
    DARK = "dark"


class ThemeColors(BaseModel):
    """Complete color palette for a theme."""

    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    surface_highlight: str
    text: str
    text_muted: str
    success: str
    warning: str
    error: str
    info: str
    border: str
    border_focus: str


class ThemeConfig(BaseModel):
    """UI theme configuration."""

    # Theme preset selection
    preset: str = "halext"  # halext, nord, solarized, dracula, gruvbox
    variant: ThemeVariant = ThemeVariant.DARK

    # Custom color overrides (applied on top of preset)
    custom: Optional[ThemeColors] = None

    # Legacy fields for backward compatibility
    primary: str = "#4C3B52"
    secondary: str = "#9B59B6"
    accent: str = "#E74C3C"
    gradient_start: str = "#4C3B52"
    gradient_end: str = "#000000"


class LayoutPreset(str, Enum):
    """Predefined layout configurations."""

    DEFAULT = "default"
    COMPACT = "compact"
    WIDE = "wide"
    FULLSCREEN_CHAT = "fullscreen_chat"


class PanelConfig(BaseModel):
    """Configuration for a UI panel."""

    visible: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    collapsed: bool = False


class UIConfig(BaseModel):
    """UI layout and panel configuration."""

    layout_preset: LayoutPreset = LayoutPreset.DEFAULT
    sidebar: PanelConfig = Field(default_factory=lambda: PanelConfig(width=32))
    context_panel: PanelConfig = Field(default_factory=lambda: PanelConfig(width=30))
    synergy_panel: PanelConfig = Field(default_factory=lambda: PanelConfig(width=18))
    remember_sizes: bool = True


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
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    knowledge_roots: list[Path] = Field(default_factory=list)
    enabled: bool = True
    description: str = ""


class NativeConfig(BaseModel):
    """Configuration for native C++ acceleration modules.

    All native features are optional and fall back to Python/NumPy when:
    - The native module isn't built (pip install without C++ toolchain)
    - A specific feature is disabled in config
    - The feature's dependency is missing (e.g., simdjson not installed)
    """

    # Master switch - disable all native acceleration
    enabled: bool = Field(
        default=True,
        description="Enable native C++ acceleration (requires build)"
    )

    # Individual feature toggles
    similarity: bool = Field(
        default=True,
        description="Use SIMD-accelerated cosine similarity"
    )
    hnsw_index: bool = Field(
        default=True,
        description="Use HNSW approximate nearest neighbor index"
    )
    quantization: bool = Field(
        default=True,
        description="Use native int8/float16 quantization"
    )
    simdjson: bool = Field(
        default=True,
        description="Use SIMD-accelerated JSON parsing (requires simdjson)"
    )
    streaming_index: bool = Field(
        default=True,
        description="Use thread-safe streaming embedding index"
    )

    # Embedding generation
    embedding_model: str = Field(
        default="embeddinggemma",
        description="Preferred Ollama embedding model"
    )
    embedding_fallback: str = Field(
        default="nomic-embed-text",
        description="Fallback model if preferred unavailable"
    )


class GeneralConfig(BaseModel):
    """General application settings."""

    refresh_interval: int = 5
    show_hidden_files: bool = False
    default_editor: str = "nvim"
    vim_navigation_enabled: bool = False
    python_executable: Optional[str] = None
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


class ObservabilityEndpointConfig(BaseModel):
    """Configuration for a monitored endpoint."""

    name: str
    url: str
    type: Optional[str] = None
    enabled: bool = True


class ObservabilityRemediationConfig(BaseModel):
    """Configuration for automatic remediation actions."""

    enabled: bool = False
    allowed_actions: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed actions: start_service, restart_service, run_afs_sync, context_burst."
        ),
    )
    allowed_services: list[str] = Field(
        default_factory=list,
        description="Service names eligible for automatic remediation.",
    )
    allowed_sync_profiles: list[str] = Field(
        default_factory=list,
        description="AFS sync profiles eligible for remediation runs.",
    )
    max_actions_per_run: int = Field(
        default=2,
        description="Maximum automatic actions per observability loop.",
    )
    cooldown_minutes: int = Field(
        default=30,
        description="Cooldown before repeating the same action on a target.",
    )
    trigger_context_burst_on_alerts: list[str] = Field(
        default_factory=lambda: ["error", "critical"],
        description="Alert severities that can trigger a context burst.",
    )
    context_burst_force: bool = Field(
        default=False,
        description="Force context burst tasks even if not due.",
    )


class ObservabilityConfig(BaseModel):
    """Configuration for observability monitoring and remediation."""

    enabled: bool = True
    check_interval_seconds: int = Field(
        default=120,
        description="Seconds between observability checks.",
    )
    endpoints: list[ObservabilityEndpointConfig] = Field(default_factory=list)
    monitor_endpoints: bool = True
    monitor_nodes: bool = True
    monitor_local_services: bool = True
    monitor_sync: bool = True
    monitor_services: bool = True
    remediation: ObservabilityRemediationConfig = Field(
        default_factory=ObservabilityRemediationConfig
    )


class LlamaCppConfig(BaseModel):
    """Configuration for the llama.cpp provider."""

    enabled: bool = True
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the llama.cpp OpenAI-compatible API (e.g. http://host:11435/v1).",
    )
    host: Optional[str] = Field(
        default=None,
        description="Host for llama.cpp when base_url is not set.",
    )
    port: Optional[int] = Field(
        default=None,
        description="Port for llama.cpp when base_url is not set.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Default model alias to use.",
    )
    api_key_env: Optional[str] = Field(
        default="LLAMACPP_API_KEY",
        description="Environment variable name that stores the API key.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="HTTP timeout in seconds.",
    )
    max_tokens: int = Field(
        default=4096,
        description="Default max tokens for llama.cpp generations.",
    )
    temperature: float = Field(
        default=0.7,
        description="Default temperature for llama.cpp generations.",
    )
    context_size: int = Field(
        default=8192,
        description="Default context size for the llama.cpp backend.",
    )


class ContextAgentModelConfig(BaseModel):
    """Model selection policy for context/background agents."""

    provider: Optional[str] = Field(
        default=None,
        description="Provider override for context agents (gemini, anthropic, openai, ollama, llamacpp).",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model override for context agents.",
    )
    rotation: list[str] = Field(
        default_factory=list,
        description="Rotation list like ['gemini:gemini-2.5-flash', 'openai:gpt-5.2-mini']",
    )
    model_tier: Optional[str] = Field(
        default=None,
        description="Task tier override for context agent prompts.",
    )
    prefer_gpu_nodes: bool = Field(
        default=False,
        description="Prefer GPU-backed Ollama nodes when available.",
    )
    prefer_remote_nodes: bool = Field(
        default=False,
        description="Prefer remote (non-local) nodes when available.",
    )


class EmbeddingDaemonConfig(BaseModel):
    """Post-embedding completion behavior for the embedding daemon."""

    post_completion_enabled: bool = Field(
        default=False,
        description="Run post-completion actions when embeddings are fully caught up.",
    )
    post_completion_mode: Literal["swarm", "coordinator"] = Field(
        default="swarm",
        description="Orchestration mode used for post-completion actions.",
    )
    post_completion_topic: str = Field(
        default="Refresh knowledge after embeddings complete.",
        description="Topic passed to the post-completion orchestration run.",
    )
    post_completion_cooldown_minutes: int = Field(
        default=240,
        description="Cooldown before another post-completion trigger can run.",
    )
    post_completion_context_burst: bool = Field(
        default=True,
        description="Request a context agent burst after embeddings complete.",
    )
    post_completion_context_force: bool = Field(
        default=True,
        description="Force context tasks to run even if not due.",
    )


class HafsConfig(BaseModel):
    """Root configuration model."""

    # Existing configuration
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    parsers: ParsersConfig = Field(default_factory=ParsersConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    context_agents: ContextAgentModelConfig = Field(default_factory=ContextAgentModelConfig)
    embedding_daemon: EmbeddingDaemonConfig = Field(default_factory=EmbeddingDaemonConfig)
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
    native: NativeConfig = Field(default_factory=NativeConfig)

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
