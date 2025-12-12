# AFS Cognitive Protocol Specification v0.2

> Building on hafs v0.1.0 Cognitive Protocol
> Reference: "Everything is Context: Agentic File System Abstraction" (2512.05470v1)

## 1. Introduction

### 1.1 Purpose

This specification defines a common protocol for the Agentic File System (AFS) cognitive layer,
ensuring cross-implementation compatibility between:

- **hafs** (Python) - TUI context manager
- **oracle-code** (TypeScript) - AI coding assistant
- Future implementations

The protocol enables:
- Immutable history logging (episodic memory)
- Cross-session learning (hivemind)
- Multi-agent consensus (council)
- Cognitive state tracking (emotions, metacognition, goals)
- Research-backed analysis modes
- Pluggable extensibility (search, storage, LLM backends)

### 1.2 Design Principles

1. **Implementation-Agnostic**: Schemas defined in JSON-compatible types
2. **Extension-First**: All major schemas include `extensions` field
3. **Append-Only History**: Immutable transaction log per paper Section IV-A
4. **Pluggable Backends**: Storage, search, and LLM are swappable
5. **Research-Backed**: Analysis modes grounded in peer-reviewed findings
6. **Security by Default**: Sensitive data filtering built into pipeline

### 1.3 Versioning

- **v0.1**: Original hafs Cognitive Protocol (Deliberative Context Loop)
- **v0.2**: This specification (History Pipeline, Analysis Modes, Event Bus)

Backward compatibility: v0.2 implementations MUST support reading v0.1 state files.

### 1.4 Research References

| Paper | Key Contribution |
|-------|-----------------|
| AFS Paper (2512.05470v1) | Core architecture, history as immutable log |
| Mind Your Tone (2510.04950v1) | Harsh prompts +4% accuracy |
| Human-AI Synergy (7799) | 17 ToM markers, collaboration metrics |
| Scaling Agents (2512.08296) | 45% threshold, architecture selection |
| AutoCommenter (3664646.3665664) | Per-rule confidence, comment categories |
| ML Code Review (7525) | Filter heuristics, edit suggestions |
| Where to Comment (3377816.3381736) | Snippet-based prediction, 74% precision |

---

## 2. History Pipeline

History is the immutable episodic memory of agent operations. Per the reference paper:

> "History records all raw interactions between users, agents, and the environment.
> Each input, output, and intermediate reasoning step is logged immutable..."

### 2.1 HistoryEntry Schema

```typescript
interface HistoryEntry {
  // Identity
  id: string;                    // ULID (26 chars, Crockford Base32)
  timestamp: string;             // ISO 8601 with timezone
  session_id: string;            // Groups related operations
  project_id?: string;           // For project-local history
  
  // Operation envelope
  operation: {
    type: OperationType;
    name: string;                // Tool name, event type, or state key
    input: Record<string, unknown>;
    output?: unknown;            // May be truncated/redacted
    duration_ms?: number;
    success: boolean;
    error?: string;
  };
  
  // Provenance
  provenance: {
    agent_id?: string;           // e.g., "general", "coder", "critic"
    model_id?: string;           // e.g., "claude-3-5-sonnet"
    parent_message_id?: string;  // Links to conversation
    parent_entry_id?: string;    // For nested operations
  };
  
  // Metadata for retrieval
  metadata: {
    tags?: string[];
    files_touched?: string[];
    token_count?: number;
    redacted?: boolean;          // True if output was sanitized
  };
  
  // Extension point
  extensions?: Record<string, unknown>;
}
```

### 2.2 OperationType Enumeration

```typescript
type OperationType =
  | "tool_call"        // MCP/built-in tool invocations
  | "agent_message"    // Agent-to-agent or agent-to-user messages
  | "user_input"       // User prompts and commands
  | "system_event"     // Internal events (session start/end, errors)
  | "cognitive_state"  // Emotional/metacognitive state snapshots
  | string;            // Extensible via plugins
```

**Cognitive State Subtypes** (for `type: "cognitive_state"`):

| `name` Value | Description |
|--------------|-------------|
| `emotions` | Mood, anxiety, satisfactions, frustrations snapshot |
| `metacognition` | Strategy, cognitive load, spin detection snapshot |
| `epistemic` | Golden facts, working facts, contradictions |
| `goals` | Goal hierarchy, focus stack, conflicts |

### 2.3 Storage Format

**Directory Structure:**
```
.context/history/
├── 2025-12-12.jsonl         # Daily operation log (append-only)
├── 2025-12-11.jsonl
├── sessions/
│   └── {session_id}.json    # Session metadata (mutable)
├── summaries/
│   └── {session_id}.md      # LLM-generated summaries
├── councils/
│   └── vote-{timestamp}-{key}.json  # Council audit logs
└── embeddings/
    └── index.db             # Vector store (optional)
```

**JSONL Format:**
- One JSON object per line
- UTF-8 encoded
- No trailing commas
- Newline (`\n`) terminated

**Rotation:**
- Daily by default: `YYYY-MM-DD.jsonl`
- Configurable: `daily | weekly | none`

**Storage Backend Protocol:**
```typescript
interface HistoryStorage {
  append(entry: HistoryEntry): Promise<void>;
  query(filters: HistoryQuery): Promise<HistoryEntry[]>;
  getSession(sessionId: string): Promise<HistoryEntry[]>;
  close(): Promise<void>;
}
```

### 2.4 Interception Protocol

All tool calls MUST be intercepted for history logging:

```typescript
interface HistoryInterceptor {
  // Called BEFORE tool execution
  onToolCallStart(
    toolName: string,
    params: Record<string, unknown>,
    provenance: Provenance
  ): Promise<string>;  // Returns entry_id
  
  // Called AFTER successful execution
  onToolCallComplete(
    entryId: string,
    result: unknown,
    durationMs: number
  ): Promise<void>;
  
  // Called on error
  onToolCallError(
    entryId: string,
    error: Error,
    durationMs: number
  ): Promise<void>;
}
```

---

## 3. Session Management

Sessions group related history entries into logical units of work.

### 3.1 SessionInfo Schema

```typescript
interface SessionInfo {
  id: string;                    // ULID
  project_id?: string;
  created_at: string;            // ISO 8601
  updated_at: string;            // ISO 8601
  
  status: "active" | "suspended" | "completed" | "aborted";
  
  // Parent session (for delegation)
  parent_session_id?: string;
  
  // Computed stats
  stats: {
    operation_count: number;
    duration_ms: number;
    files_modified: string[];
    tools_used: string[];
  };
  
  // LLM-generated summary (populated on complete)
  summary?: {
    title?: string;
    body?: string;
  };
  
  extensions?: Record<string, unknown>;
}
```

### 3.2 Session Lifecycle

```
create() ─────► active
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
suspended    completed     aborted
    │            │            │
    │            ▼            ▼
    │      [summarize]    [log reason]
    │
    └──► resume() ──► active
```

**State Transitions:**
- `create()` → `active`: New session
- `suspend()` → `suspended`: User away, context switch
- `resume(id)` → `active`: Continue suspended session
- `complete()` → `completed`: Normal end, triggers summarization
- `abort(reason)` → `aborted`: Error, user cancel, timeout

### 3.3 SessionSummary Schema

```typescript
interface SessionSummary {
  session_id: string;
  project_id?: string;
  created_at: string;
  
  // LLM-generated content
  summary: string;
  
  // Extracted entities (for knowledge graph)
  entities?: Array<{
    name: string;
    type: string;      // "file" | "function" | "concept" | "person" | custom
    mentions: number;
  }>;
  
  // Statistics
  stats: {
    operation_count: number;
    duration_ms: number;
    files_modified: string[];
    tools_used: string[];
  };
  
  // Embedding (dimension flexible per implementation)
  embedding?: number[];
  embedding_model?: string;
  
  extensions?: Record<string, unknown>;
}
```

---

## 4. Cognitive State Protocol

Cognitive state entries capture the agent's internal state for Theory of Mind
and self-monitoring.

### 4.1 Emotions Schema

```typescript
interface EmotionsState {
  mood: {
    current: string;           // "neutral" | "focused" | "frustrated" | etc.
    intensity: number;         // 0-1
    valence: number;           // -1 to 1 (negative to positive)
  };
  
  anxiety: {
    level: number;             // 0-1
    sources: string[];         // What's causing anxiety
  };
  
  satisfactions: Array<{
    id: string;
    description: string;
    intensity: number;         // 1-10
    timestamp: string;
  }>;
  
  frustrations: Array<{
    id: string;
    description: string;
    intensity: number;         // 1-10
    resolved: boolean;
    timestamp: string;
  }>;
  
  mood_history: Array<{
    mood: string;
    timestamp: string;
    trigger?: string;
  }>;
}
```

### 4.2 Metacognition Schema

```typescript
interface MetacognitionState {
  current_strategy: 
    | "incremental"
    | "divide_and_conquer"
    | "depth_first"
    | "breadth_first"
    | "research_first"
    | "prototype";
  
  strategy_effectiveness: number;  // 0-1
  
  progress_status: "making_progress" | "spinning" | "blocked";
  
  spin_detection: {
    recent_actions: string[];      // Hashes of recent actions
    similar_action_count: number;
    spinning_threshold: number;    // Default: 4
  };
  
  cognitive_load: {
    current: number;               // 0-1 (percentage)
    items_in_focus: number;        // Miller's Law: ~7 max
  };
  
  help_seeking: {
    current_uncertainty: number;   // 0-1
    consecutive_failures: number;
    should_ask_user: boolean;
  };
  
  flow_state: boolean;             // Autonomous operation mode
  
  self_corrections: Array<{
    what: string;
    when: string;
    why: string;
    outcome: string;
  }>;
}
```

### 4.3 Goals Schema

```typescript
interface GoalsState {
  primary_goal?: {
    id: string;
    description: string;
    user_stated: string;
    status: "pending" | "in_progress" | "completed" | "blocked" | "abandoned";
    progress: number;              // 0-1
    success_criteria: string[];
    constraints: string[];
  };
  
  subgoals: Array<{
    id: string;
    parent_id: string;
    description: string;
    status: string;
    progress: number;
    dependencies: string[];
  }>;
  
  instrumental_goals: Array<{
    id: string;
    description: string;
    supports: string[];            // Goal IDs this enables
    status: string;
  }>;
  
  goal_stack: string[];            // Active focus (top = current)
  
  conflicts: Array<{
    type: string;
    goal_a: string;
    goal_b: string;
    resolution?: string;
  }>;
  
  last_updated: string;
}
```

### 4.4 Epistemic Schema

```typescript
interface EpistemicState {
  golden_facts: Array<{
    id: string;
    fact: string;
    confidence: number;            // 1.0 for golden
    source: string;
    verified_at: string;
  }>;
  
  working_facts: Array<{
    id: string;
    fact: string;
    confidence: number;            // 0-1
    source: string;
  }>;
  
  contradictions: Array<{
    fact_a: string;
    fact_b: string;
    detected_at: string;
    resolution?: string;
  }>;
  
  max_golden: number;              // Default: 10
  max_working: number;             // Default: 100
}
```

### 4.5 Cognitive State Frequency

```typescript
interface CognitiveStateConfig {
  // Frequency mode
  mode: "on_change" | "throttled" | "delta";  // Default: "delta"
  
  // Throttle settings (when mode = "throttled")
  min_interval_ms?: number;  // Default: 60000 (1 minute)
  
  // Delta settings (when mode = "delta")
  delta_threshold?: {
    mood_change: boolean;           // Log on mood change
    anxiety_delta: number;          // Log if anxiety changes by this amount (default: 0.1)
    cognitive_load_delta: number;   // Log if load changes by this amount (default: 0.2)
    strategy_change: boolean;       // Log on strategy change
    goal_status_change: boolean;    // Log on goal completion/blocking
  };
}
```

---

## 5. Analysis Mode Protocol

Analysis modes provide structured evaluation of agent behavior, collaboration quality,
and system performance. Modes integrate findings from peer-reviewed research.

### 5.1 Mode Enumeration

```typescript
type AnalysisMode = 
  | "none"           // Standard operation, no analysis overlay
  | "eval"           // Prompt/response quality evaluation
  | "tom"            // Theory of Mind marker detection
  | "metrics"        // Coordination efficiency and scaling metrics
  | "critic"         // Adaptive harsh criticism
  | "emotional"      // Emotional valence and cognitive load
  | "synergy"        // Human-AI collaboration quality (future)
  | "review"         // Google-style code review
  | "documentation"  // Comment placement analysis
```

### 5.2 Mode Specifications

#### 5.2.1 `eval` - Evaluation Mode

Analyzes prompt and response quality for optimization.

```typescript
interface EvalMetrics {
  prompt_quality: {
    clarity: number;           // 0-1
    specificity: number;       // 0-1
    context_sufficiency: number;
    ambiguity_count: number;
  };
  
  response_quality: {
    correctness: number;       // 0-1 (when verifiable)
    helpfulness: number;       // 0-1
    completeness: number;      // 0-1
    reasoning_quality: number; // 0-5 scale
  };
  
  information_gain: number;    // ΔI = ½ log(Var[Y|s_pre] / Var[Y|s_post])
}
```

#### 5.2.2 `tom` - Theory of Mind Mode

Tracks ToM markers per "Quantifying Human-AI Synergy" paper (7799).

**ToM Marker Types (17 indicators):**

```typescript
type ToMMarkerType =
  // High ToM Indicators
  | "perspective_taking"       // "from your perspective", "if I were you"
  | "goal_inference"           // "your goal is", "you're trying to"
  | "knowledge_gap_detection"  // "you might not know", "I should mention"
  | "communication_repair"     // "let me clarify", "in other words"
  | "confirmation_seeking"     // "is that correct?", "does that make sense?"
  | "mental_state_attribution" // "as an AI", "given your capabilities"
  | "plan_coordination"        // "let's work together", "can you handle"
  | "belief_tracking"          // Inference of agent beliefs about state
  | "explanatory_dialogue"     // Context about knowledge level
  | "build_on_ideas"           // Elaboration showing understanding
  | "reference_back"           // Explicit acknowledgment of prior thoughts
  | "challenge_disagree"       // "are you sure", "I disagree"
  | "epistemic_markers"        // Language revealing knowledge states
  | "justification_requests"   // Questions about reasoning processes
  | "metacognitive_references" // Discussion of thinking processes
  | "conversational_adaptation" // Real-time adjustment to style
  // Low ToM Indicators (negative markers)
  | "irrelevant_sharing"       // Failing to detect knowledge asymmetries
  | "assumed_context"          // Without establishing it
  | "capability_misunderstanding" // Treating AI incorrectly
```

**ToM Analysis Output:**

```typescript
interface ToMAnalysis {
  markers_detected: Array<{
    type: ToMMarkerType;
    text: string;
    position: number;
    confidence: number;
  }>;
  
  // Trait-level (stable across session)
  trait_tom_score: number;     // 0-1, user-level mean
  
  // Dynamic (per-turn deviation)
  dynamic_tom_deviation: number;  // ToM_ij - ToM_i(-j)
  
  // Correlation with outcomes (per Synergy paper β=0.27***)
  predicted_response_quality: number;
}
```

#### 5.2.3 `metrics` - Coordination Metrics Mode

Tracks multi-agent coordination per "Scaling Agent Systems" paper (2512.08296).

**Architecture Types:**

```typescript
type AgentArchitecture = 
  | "single"       // One agent, sequential
  | "independent"  // Multiple agents, no inter-agent communication
  | "centralized"  // Orchestrator coordinates worker agents
  | "decentralized" // All agents can communicate with each other
  | "hybrid"       // Centralized + peer communication
```

**Scaling Metrics:**

```typescript
interface ScalingMetrics {
  // Primary metrics
  coordination_overhead: number;  // O% = (T_MAS - T_SAS) / T_SAS × 100%
  message_density: number;        // Inter-agent messages per turn
  redundancy_rate: number;        // Cosine similarity of agent outputs
  coordination_efficiency: number; // E_c = S / (T / T_SAS)
  error_amplification: number;    // A_e = E_MAS / E_SAS
  
  // Decision support
  baseline_accuracy: number;      // P_SA - single agent baseline
  task_tool_count: number;        // T - number of tools available
  task_decomposability: number;   // D - parallelizability score 0-1
  
  // Thresholds (from paper)
  should_use_multi_agent: boolean; // P_SA < 0.45 AND T <= 4
  recommended_architecture: AgentArchitecture;
  max_effective_agents: number;   // Usually 3-4
  
  // Error analysis by architecture
  error_rates: {
    logical_contradiction: number;
    numerical_drift: number;
    context_omission: number;
    coordination_failure: number;
  };
}
```

**Architecture Selection Formula:**

```
IF baseline_accuracy > 0.45:
    RECOMMEND single
ELIF task.decomposability > 0.3 AND task.output_structured:
    RECOMMEND centralized (error containment: 4.4× vs 17.2×)
ELIF task.requires_exploration OR task.high_entropy:
    RECOMMEND decentralized (+9.2% on dynamic tasks)
ELIF need_diversity WITHOUT coordination:
    RECOMMEND independent (warn: 17.2× error amplification)
ELSE:
    RECOMMEND hybrid (warn: 515% overhead)
```

#### 5.2.4 `critic` - Adaptive Harsh Critic Mode

Applies direct/challenging tone per "Mind Your Tone" findings (2510.04950v1).

**Adaptive Tone Configuration:**

```typescript
interface AdaptiveCriticConfig {
  // Default: harsh (per Mind Your Tone findings: +4% accuracy)
  default_tone: "harsh";
  
  // Automatic downgrade rules
  downgrade_rules: {
    // If anxiety exceeds threshold, reduce harshness
    anxiety_threshold: 0.7;
    anxiety_downgrade_to: "direct";
    
    // If consecutive frustrations, reduce further
    frustration_count_threshold: 3;
    frustration_downgrade_to: "neutral";
    
    // Manual override always available
    allow_user_escalation: true;
    allow_user_override: true;
  };
  
  // Gradual escalation when stable
  escalation_rules: {
    // After N successful iterations without anxiety spike
    stable_iterations_for_escalation: 5;
    escalation_path: ["neutral", "direct", "challenging", "harsh"];
  };
}
```

**Tone State Machine:**

```
harsh ──(anxiety > 0.7)──► direct ──(frustrations >= 3)──► neutral
  ▲                          ▲                               │
  └──(stable × 5)────────────┴───────(stable × 5)───────────┘
```

**Tone Prefixes (from paper):**

| Level | Prefix Example |
|-------|---------------|
| neutral | (none) |
| direct | "Identify issues in this:" |
| challenging | "I doubt this is correct. Find the problems:" |
| harsh | "This looks wrong. Point out every flaw:" |

**Critic Output Format:**

```typescript
interface CriticReview {
  severity: "critical" | "major" | "minor" | "nitpick";
  aspect: string;
  location?: { file: string; line?: number };
  issue: string;
  suggestion: string;
  confidence: number;
}
```

#### 5.2.5 `emotional` - Emotional Valence Mode

Tracks agent emotional state per Cognitive Protocol.

```typescript
interface EmotionalAnalysis {
  mood: {
    current: "neutral" | "focused" | "frustrated" | "satisfied" | "anxious";
    intensity: number;  // 0-1
    valence: number;    // -1 to +1
  };
  
  anxiety: {
    level: number;      // 0-1
    sources: string[];
    mitigation_suggestions: string[];
  };
  
  // Active emotional entries
  fears: Array<{ id: string; description: string; intensity: number }>;
  satisfactions: Array<{ id: string; description: string; intensity: number }>;
  frustrations: Array<{ id: string; description: string; intensity: number; resolved: boolean }>;
  
  // Thresholds
  high_anxiety_threshold: 0.7;  // Trigger caution mode
  frustration_spinning_threshold: 3;  // Consecutive frustrations
}
```

#### 5.2.6 `synergy` - Human-AI Synergy Mode (Future)

Measures collaboration quality per "Quantifying Human-AI Synergy" paper.

**Note:** Implementation deferred until historical data available for estimation.

```typescript
interface SynergyAnalysis {
  // Core synergy calculation
  human_solo_ability: number;     // θ_i^human (estimated from context)
  ai_collaborative_ability: number; // κ_m^AI
  combined_performance: number;   // κ_i,AI^total
  ai_boost: number;               // κ_i,AI^total - θ_i^human
  
  // Decomposed abilities
  user_collaborative_ability: number; // κ_i^human
  
  // Task difficulty interaction
  task_difficulty_rank: number;
  expected_ai_boost_by_difficulty: number;
  
  // Equalizing effect
  relative_boost: number;  // Higher for lower-ability users
}
```

#### 5.2.7 `review` - Code Review Mode

Based on AutoCommenter (3664646.3665664) and ML Code Review (7525) papers.

```typescript
interface ReviewModeConfig {
  // Comment categories (from AutoCommenter paper)
  categories: {
    formatting: { enabled: boolean; priority: "low" };
    naming: { enabled: boolean; priority: "medium" };
    documentation: { enabled: boolean; priority: "medium" };
    language_features: { enabled: boolean; priority: "high" };
    code_idioms: { enabled: boolean; priority: "high" };
    best_practices: { enabled: boolean; priority: "high" };
  };
  
  // Per-category confidence thresholds (not global)
  confidence_thresholds: Record<string, number>;
  
  // Filter heuristics (from ML code review paper)
  filters: {
    max_lines_from_comment: 5;       // Reject edits >5 lines away
    reject_todo_additions: true;      // Filter "promise to fix later"
    reject_delete_only: true;         // Protect user trust
    reject_unchanged_lines: true;     // Only comment on changed code
    reject_file_level_comments: true; // Require line reference
  };
  
  // Reviewer preview (63% of bad suggestions caught)
  reviewer_preview: boolean;
  
  // Style vs substance
  style_nits_enabled: boolean;  // Line limits, whitespace, etc.
}

interface ReviewComment {
  category: string;
  severity: "critical" | "major" | "minor" | "nit";
  location: { file: string; line: number; snippet?: string };
  comment: string;
  suggested_edit?: string;
  best_practice_url?: string;  // Reference to guideline
  confidence: number;
  automatable: boolean;        // Could linter handle this?
}
```

#### 5.2.8 `documentation` - Documentation Analysis Mode

Based on "Where should I comment my code?" paper (3377816.3381736).

```typescript
interface DocumentationModeConfig {
  // Snippet detection
  snippet_detection: {
    delimiter: "blank_line";        // How to segment code
    max_snippet_lines: 30;          // Split large snippets
  };
  
  // Prediction thresholds (from paper: 74% precision, 13% recall)
  thresholds: {
    min_precision: 0.7;             // Only suggest with high confidence
    // Note: low recall acceptable - better to miss some than suggest wrong
  };
  
  // What to analyze
  analyze: {
    missing_comments: boolean;      // Predict where comments should be
    comment_quality: boolean;       // Evaluate existing comment quality
    comment_placement: boolean;     // Is comment in right location?
  };
  
  // Integration with critic
  escalate_to_critic: boolean;     // Pass findings to critic mode
}

interface DocumentationAnalysis {
  snippets_analyzed: number;
  snippets_needing_comments: Array<{
    location: { file: string; start_line: number; end_line: number };
    confidence: number;
    reason: string;
    suggested_comment_location: "top" | "inline" | "end";
  }>;
  
  existing_comment_issues: Array<{
    location: { file: string; line: number };
    issue: "outdated" | "unclear" | "wrong_location" | "too_verbose";
    suggestion: string;
  }>;
}
```

### 5.3 Analysis Triggers

Automatic invocation of analysis based on context.

**Trigger Schema:**

```typescript
interface AnalysisTrigger {
  id: string;
  name: string;
  description: string;
  
  // Condition
  condition: {
    type: "threshold" | "pattern" | "count" | "time";
    metric?: string;
    threshold?: number;
    pattern?: string;  // Regex
    count?: number;
    within_ms?: number;
  };
  
  // Action
  action: {
    mode: AnalysisMode;
    agent?: string;           // Subagent to spawn
    auto_accept?: boolean;    // Skip confirmation
    priority: "low" | "medium" | "high" | "critical";
  };
  
  // Cooldown
  cooldown_ms?: number;
  max_triggers_per_session?: number;
}
```

**Default Triggers (13 total):**

| ID | Condition | Action | Priority |
|----|-----------|--------|----------|
| `spinning-critic` | progress_status == "spinning" | critic | high |
| `edits-without-tests` | edit_count >= 3 AND test_count == 0 | eval | medium |
| `new-territory-explore` | unfamiliar_code_detected | tom | low |
| `contradiction-debate` | epistemic.contradictions.length > 0 | council | high |
| `high-anxiety-caution` | anxiety_level > 0.7 | emotional | high |
| `consecutive-failures` | consecutive_failures >= 3 | metrics | high |
| `critical-unknowns` | epistemic.unknowns.filter(critical).length > 0 | synergy | medium |
| `high-cognitive-load` | cognitive_load > 0.8 | emotional | medium |
| `low-strategy-effectiveness` | strategy_effectiveness < 0.3 | metrics | medium |
| `tool-repetition` | same_tool_count >= 5 | critic | low |
| `baseline-too-high` | baseline_accuracy > 0.45 AND agent_count > 1 | metrics | medium |
| `error-amplification-warning` | error_amplification > 10 | metrics | high |
| `overhead-exceeded` | coordination_overhead > 150% AND tool_count > 12 | metrics | medium |

### 5.4 Analysis Gate Modes

Control how triggers are handled:

```typescript
type AnalysisGateMode = 
  | "confirm-all"   // All triggers require user confirmation (default)
  | "auto-accept"   // Flow-friendly automatic acceptance
  | "auto-deny"     // Silently ignore all triggers
  | "selective"     // Per-trigger configuration
```

### 5.5 LLM Backend Selection for Analysis

Analysis can use different backends based on cost/privacy needs:

```typescript
interface AnalysisBackendConfig {
  // Backend selection per analysis type
  backends: {
    eval: string;              // Default: "default" (cloud)
    tom: string;               // Default: "default"
    metrics: string;           // Default: "local" (no LLM needed)
    critic: string;            // Default: "default"
    emotional: string;         // Default: "ollama" (privacy)
    synergy: string;           // Default: "default"
    review: string;            // Default: "default"
    documentation: string;     // Default: "ollama" (can be local)
  };
  
  // Local model configuration
  local: {
    ollama?: {
      host: string;            // Default: "http://localhost:11434"
      model: string;           // Default: "llama3"
      embed_model: string;     // Default: "nomic-embed-text"
    };
    llamacpp?: {
      host: string;            // Default: "http://localhost:8080"
    };
  };
  
  // Fallback behavior
  fallback_to_cloud: boolean;  // Default: true
  fallback_delay_ms: number;   // Wait before fallback: 5000
}
```

---

## 6. Hivemind Protocol

Cross-session learning through persistent knowledge entries.

### 6.1 HivemindEntry Schema

```typescript
interface HivemindEntry {
  id: string;                      // ULID
  
  category: "fear" | "satisfaction" | "knowledge" | "decision" | "preference";
  scope: "project" | "global";
  
  key: string;                     // Short identifier
  value: string;                   // The actual content
  confidence: number;              // 0-1
  
  status: "active" | "decaying" | "golden" | "contested";
  
  source: {
    session_id: string;
    agent_role: string;
    timestamp: string;
    promotion_reason: string;
  };
  
  decay: {
    last_accessed: string;
    access_count: number;
    decay_rate: number;            // Per day (0 = no decay)
    expires_at?: string;
  };
  
  metadata: {
    related_entries?: string[];
    tags?: string[];
    original_entry_id?: string;
  };
  
  // Conflict resolution
  contested?: {
    reason: string;
    alternative_value: string;
    council_session_id?: string;
  };
  
  // Golden promotion
  golden?: {
    promoted_at: string;
    promoted_by: string;
    council_session_id?: string;
  };
  
  // Cross-project sync
  sync?: {
    source_project?: string;
    synced_to?: string[];
    last_sync?: string;
  };
  
  extensions?: Record<string, unknown>;
}
```

### 6.2 Decay Protocol

**Formula:**
```
effective_strength = confidence × (1 - decay_rate)^days_since_access
```

**Default Decay Rates:**

| Category | Rate | Notes |
|----------|------|-------|
| fear | 0.10 | 10% per day |
| satisfaction | 0.10 | 10% per day |
| knowledge | 0.00 | No decay (facts are permanent) |
| decision | 0.05 | 5% per day |
| preference | 0.00 | No decay |

**Status Transitions:**
- `active` → `decaying`: When within warning threshold
- `decaying` → expired (removed): When `effective_strength < 0.1`
- `decaying` → `contested`: When nominated for golden promotion
- `contested` → `golden` | `active`: After council vote

### 6.3 Storage Locations

**Directory Structure:**

```
~/.context/
├── global/
│   ├── oracle-code/           # App-specific
│   │   ├── hivemind/
│   │   │   ├── manifest.json
│   │   │   ├── fears.json
│   │   │   ├── satisfactions.json
│   │   │   ├── knowledge.json
│   │   │   ├── decisions.json
│   │   │   └── preferences.json
│   │   └── history/
│   ├── hafs/                   # App-specific
│   │   ├── hivemind/
│   │   └── history/
│   └── shared/                 # Cross-app
│       ├── project-groups.json
│       ├── hivemind/
│       │   ├── knowledge.json
│       │   ├── preferences.json
│       │   └── decisions.json
│       └── groups/
│           ├── zelda/          # yaze + oracle-of-secrets + usdasm
│           │   └── hivemind/
│           ├── agentic/        # hafs + oracle-code + halext-code
│           │   └── hivemind/
│           └── halext-web/     # halext-org + halext-server
│               └── hivemind/
```

**Project-Local:**
```
.context/hivemind/
├── manifest.json
├── fears.json
├── satisfactions.json
├── knowledge.json
├── decisions.json
├── preferences.json
├── pending.json
└── councils.json
```

### 6.4 Project Groups Configuration

Location: `~/.context/global/shared/project-groups.json`

```typescript
interface ProjectGroupsConfig {
  version: string;
  
  groups: Record<string, {
    description: string;
    projects: string[];
    shared_categories: Array<"knowledge" | "decision" | "preference">;
  }>;
  
  project_paths: Record<string, string>;
}
```

### 6.5 Cross-Project Sync Protocol

- Only `knowledge`, `preference`, and `decision` categories sync to shared
- `fear` and `satisfaction` remain app-specific (context-dependent)
- Sync modes: `push | pull | bidirectional`
- Conflict resolution: `skip | replace | council`

---

## 7. Council Protocol

Multi-agent consensus for conflict resolution and promotions.

### 7.1 CouncilSession Schema

```typescript
interface CouncilSession {
  id: string;                      // ULID
  
  purpose: "conflict" | "decay_promotion" | "global_promotion";
  entry_key: string;
  contest_reason: string;
  
  current_value: string;
  proposed_value: string;
  
  votes: Array<{
    agent_role: string;
    vote: "approve" | "reject" | "abstain";
    confidence: number;            // 0-1, weights the vote
    rationale: string;
    timestamp: string;
  }>;
  
  config: {
    council_size: number;          // Default: 3
    quorum: number;                // Default: 2
    threshold: number;             // Default: 0.67
    debate_on_tie: boolean;        // Default: true
    council_agents: string[];      // Default: ["explore", "critic", "general"]
  };
  
  status: "voting" | "approved" | "rejected" | "tie" | "debating";
  
  result?: {
    final_decision: "approve" | "reject";
    confidence: number;
    rationale: string;
  };
  
  debate?: {
    rounds: Array<{
      round: number;
      pro_argument: string;
      con_argument: string;
      pro_agent: string;
      con_agent: string;
    }>;
    arbitrator_agent: string;
    final_synthesis?: string;
  };
  
  audit: {
    initiated_by: string;
    initiated_at: string;
    completed_at?: string;
    log_path?: string;
  };
  
  timestamp: string;
}
```

### 7.2 Voting Protocol

1. **Session Creation**: Generate ULID, persist to `councils.json`
2. **Agent Spawning**: For each agent in `council_agents`
3. **Vote Calculation**:
   ```
   approve_weight = Σ(approve_votes × confidence)
   reject_weight = Σ(reject_votes × confidence)
   approval_ratio = approve_weight / (approve_weight + reject_weight)
   ```
4. **Decision**:
   - If `approval_ratio >= threshold`: APPROVED
   - If `approval_ratio <= (1 - threshold)`: REJECTED
   - Otherwise: TIE → trigger debate if configured

### 7.3 Architecture Recommendations

When metrics mode detects suboptimal architecture:
- Recommendations presented with council-style confirmation
- User can approve, reject, or request more info
- Not automatic switching

### 7.4 Audit Trail

**Log Path**: `.context/history/councils/vote-{ISO-timestamp}-{sanitized-key}.json`

**Retention**: 90 days default (configurable)

---

## 8. Event Bus Protocol

Standard events published by each subsystem for cross-component communication.

### 8.1 Event Naming Convention

```
{subsystem}.{entity}.{action}
```

Examples: `history.entry.created`, `hivemind.entry.promoted_golden`

### 8.2 Transport Abstraction

```typescript
interface IPCTransport {
  send(event: BusEvent): Promise<void>;
  subscribe(pattern: string, handler: EventHandler): Unsubscribe;
  isConnected(): boolean;
}

// Primary: HTTP (oracle-code has Hono server on :4096)
class HTTPTransport implements IPCTransport {
  constructor(baseUrl: string = "http://localhost:4096");
}

// Optional: Unix sockets for high-performance
class UnixSocketTransport implements IPCTransport {
  constructor(socketPath?: string);
  // Linux: /run/user/$UID/oracle-code.sock
  // macOS: ~/Library/Caches/oracle-code/oracle-code.sock
}
```

### 8.3 Event Categories

**History Events:**
| Event | Payload |
|-------|---------|
| `history.entry.created` | `{ entry: HistoryEntry }` |
| `history.session.started` | `{ session: SessionInfo }` |
| `history.session.completed` | `{ session: SessionInfo, summary: SessionSummary }` |
| `history.session.aborted` | `{ session: SessionInfo, reason: string }` |

**Hivemind Events:**
| Event | Payload |
|-------|---------|
| `hivemind.entry.added` | `{ entry: HivemindEntry, scope: string }` |
| `hivemind.entry.updated` | `{ entry: HivemindEntry, changes: string[] }` |
| `hivemind.entry.removed` | `{ entryId: string, reason: string }` |
| `hivemind.entry.promoted_golden` | `{ entry: HivemindEntry }` |
| `hivemind.decay.entry_expired` | `{ entry: HivemindEntry }` |

**Council Events:**
| Event | Payload |
|-------|---------|
| `council.session.created` | `{ session: CouncilSession }` |
| `council.vote.cast` | `{ sessionId: string, vote: Vote }` |
| `council.session.resolved` | `{ session: CouncilSession, result: Result }` |

**Cognitive Events:**
| Event | Payload |
|-------|---------|
| `cognitive.emotions.updated` | `{ state: EmotionsState }` |
| `cognitive.metacognition.updated` | `{ state: MetacognitionState }` |
| `cognitive.spin_detected` | `{ actionCount: number }` |
| `cognitive.flow_state.entered` | `{ conditions: string[] }` |
| `cognitive.flow_state.exited` | `{ reason: string }` |

**Analysis Events:**
| Event | Payload |
|-------|---------|
| `analysis.mode.activated` | `{ mode: AnalysisMode }` |
| `analysis.trigger.fired` | `{ trigger: AnalysisTrigger }` |
| `analysis.critic.tone_changed` | `{ from: string, to: string, reason: string }` |

**Search Events:**
| Event | Payload |
|-------|---------|
| `search.query.executed` | `{ query: SearchQuery, resultCount: number }` |
| `search.result.selected` | `{ queryId: string, source: string, rank: number }` |

### 8.4 Per-Event Configuration

Events can be individually configured:

```typescript
interface EventConfig {
  // Which events to sync cross-process
  sync: {
    [eventPattern: string]: {
      enabled: boolean;
      transport: "http" | "unix" | "file";
      priority: "realtime" | "batch";
    };
  };
}
```

Default: Hivemind and cognitive events sync, history events local-only.

---

## 9. Search Protocol

Unified search across multiple backends with performance-based weighting.

### 9.1 SearchQuery Schema

```typescript
interface SearchQuery {
  id: string;
  query: string;
  
  modes: {
    ripgrep?: boolean;
    semantic?: boolean;
    fuzzy?: boolean;
    knowledge_graph?: boolean;
  };
  
  filters?: {
    project_id?: string;
    date_range?: { start: string; end: string };
    file_patterns?: string[];
    tags?: string[];
    operation_types?: string[];
  };
  
  weights?: {
    ripgrep?: number;       // Default: 1.0
    semantic?: number;      // Default: 1.0
    fuzzy?: number;         // Default: 0.5
    knowledge_graph?: number; // Default: 0.8
  };
  
  limit?: number;           // Default: 20
}
```

### 9.2 SearchResult Schema

```typescript
interface SearchResult {
  source: string;
  score: number;            // 0-1 normalized
  weighted_score: number;
  
  data: {
    file_path?: string;
    line_number?: number;
    content?: string;
    session_id?: string;
    entry_id?: string;
    entity?: { name: string; type: string };
    [key: string]: unknown;
  };
}
```

### 9.3 SearchProvider Interface

```typescript
interface SearchProvider {
  readonly name: string;
  search(query: SearchQuery): Promise<SearchResult[]>;
  isAvailable?(): Promise<boolean>;
}
```

### 9.4 Performance Tracking

**Metrics File**: `.context/metrics/search-performance.json`

```typescript
interface SearchMetrics {
  queries: Array<{
    id: string;
    query: string;
    modes_used: string[];
    results_count: number;
    latency_ms: number;
    user_selected_result?: { source: string; rank: number };
    timestamp: string;
  }>;
  
  aggregates: {
    [provider: string]: {
      query_count: number;
      avg_latency_ms: number;
      selection_rate: number;
      avg_result_rank: number;
    };
  };
}
```

---

## 10. LLM Backend Protocol

Pluggable LLM interface for summarization, embedding, and generation.

### 10.1 LLMBackend Interface

```typescript
interface LLMBackend {
  readonly name: string;
  
  generate(
    prompt: string,
    options?: GenerateOptions
  ): Promise<string | AsyncIterator<string>>;
  
  embed(text: string): Promise<number[]>;
  
  isAvailable(): Promise<boolean>;
}

interface GenerateOptions {
  system?: string;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}
```

### 10.2 Embedding Interface

Dimension is flexible - varies by model:

```typescript
interface Embeddable {
  embedding?: number[];           // Dimension varies by model
  embedding_model?: string;       // e.g., "nomic-embed-text"
  embedding_timestamp?: string;   // When embedding was generated
}
```

### 10.3 Reference Implementations

**OllamaBackend:**
```typescript
class OllamaBackend implements LLMBackend {
  readonly name = "ollama";
  
  constructor(options: {
    host?: string;           // Default: "http://localhost:11434"
    model?: string;          // Default: "llama3"
    embed_model?: string;    // Default: "nomic-embed-text"
  });
}
```

**LlamaCppBackend (Future):**
```typescript
class LlamaCppBackend implements LLMBackend {
  readonly name = "llamacpp";
  
  constructor(options: {
    host?: string;           // Default: "http://localhost:8080"
    model_path?: string;
  });
}
```

### 10.4 Backend Registry

```typescript
interface LLMRegistry {
  register(name: string, backend: LLMBackend): void;
  get(name?: string): LLMBackend | undefined;
  list(): string[];
  setDefault(name: string): void;
}
```

---

## 11. Sanitization Protocol

Prevent sensitive data from leaking into history, logs, or commits.

### 11.1 Sensitive Data Patterns

**MUST redact:**

| Category | Patterns |
|----------|----------|
| API Keys | `/[A-Za-z0-9_-]{20,}/` in env-like contexts |
| Env Files | `*.env`, `*.env.*`, `.env*` |
| Credentials | `password=`, `secret=`, `token=`, `api_key=` |
| Private Keys | `-----BEGIN.*PRIVATE KEY-----` |
| PII Patterns | Email addresses, phone numbers (configurable) |

### 11.2 Sanitization Points

1. **History Logging**: Before writing `HistoryEntry.operation.output`
2. **Tool Output Display**: Before rendering to user
3. **Summary Generation**: Before sending to LLM
4. **Export/Transfer**: Before writing to file

### 11.3 Sanitizer Interface

```typescript
interface Sanitizer {
  sanitize(
    value: unknown,
    context: SanitizationContext
  ): { value: unknown; redacted: boolean };
  
  shouldExcludeFile(path: string): boolean;
  
  addPattern(name: string, pattern: RegExp, replacement?: string): void;
}

interface SanitizationContext {
  source: "history" | "output" | "summary" | "export";
  tool_name?: string;
  file_path?: string;
}
```

**Default Replacements:**

| Pattern | Replacement |
|---------|-------------|
| API key | `[REDACTED_API_KEY]` |
| Password | `[REDACTED_PASSWORD]` |
| Private key | `[REDACTED_PRIVATE_KEY]` |
| Email | `[REDACTED_EMAIL]` |

---

## 12. Plugin Protocol

Extensibility through typed plugin interfaces.

### 12.1 Base Plugin Interface

```typescript
interface Plugin {
  readonly name: string;
  readonly version: string;        // SemVer
  
  activate(app: AppContext): Promise<void>;
  deactivate(): Promise<void>;
}
```

### 12.2 Plugin Types

| Type | Provides |
|------|----------|
| `BackendPlugin` | LLM backend implementation |
| `SearchPlugin` | Search provider implementation |
| `StoragePlugin` | History storage backend |
| `ParserPlugin` | Log/data parser (hafs) |
| `WidgetPlugin` | UI component (hafs) |
| `SanitizerPlugin` | Custom sanitization rules |
| `AnalysisPlugin` | Custom analysis mode |

### 12.3 Discovery Mechanisms

**Python (hafs):**
- Entry points: `hafs.plugins`
- Directory: `~/.config/hafs/plugins/`

**TypeScript (oracle-code):**
- Package exports
- Directory: `~/.config/oracle-code/plugins/`

---

## 13. Configuration Schema

### 13.1 History Config

```typescript
interface HistoryConfig {
  storage_backend: "file" | "sqlite" | string;
  rotation: "daily" | "weekly" | "none";
  retention_days: number;          // 0 = forever
  interception_enabled: boolean;
  max_output_length: number;       // Truncate large outputs
}
```

### 13.2 Hivemind Config

```typescript
interface HivemindConfig {
  global_enabled: boolean;
  
  decay: {
    default_rates: Record<string, number>;
    check_interval_ms: number;
    warning_threshold_days: number;
    golden_exempt: boolean;
    preferences_exempt: boolean;
  };
  
  council: {
    council_size: number;
    quorum: number;
    threshold: number;
    debate_on_tie: boolean;
    council_agents: string[];
    max_debate_rounds: number;
  };
}
```

### 13.3 Analysis Config

```typescript
interface AnalysisConfig {
  default_mode: AnalysisMode;
  gate_mode: AnalysisGateMode;
  
  triggers: {
    enabled: boolean;
    custom_triggers?: AnalysisTrigger[];
    disabled_triggers?: string[];
  };
  
  critic: AdaptiveCriticConfig;
  review: ReviewModeConfig;
  documentation: DocumentationModeConfig;
  
  backends: AnalysisBackendConfig;
}
```

### 13.4 Search Config

```typescript
interface SearchConfig {
  default_modes: string[];
  default_weights: Record<string, number>;
  providers: string[];
  metrics_enabled: boolean;
}
```

### 13.5 LLM Config

```typescript
interface LLMConfig {
  preferred_backend: string;
  
  backends: {
    ollama?: {
      host: string;
      model: string;
      embed_model: string;
    };
  };
}
```

### 13.6 Sanitization Config

```typescript
interface SanitizationConfig {
  enabled: boolean;
  
  patterns: {
    api_keys: boolean;
    credentials: boolean;
    pii: boolean;
  };
  
  custom_patterns: Array<{
    name: string;
    pattern: string;
    replacement: string;
  }>;
  
  excluded_tools: string[];
}
```

---

## 14. Appendix

### A. ULID Specification

ULIDs (Universally Unique Lexicographically Sortable Identifiers):
- 26 characters, Crockford Base32
- First 10 chars: Timestamp (48-bit, milliseconds)
- Last 16 chars: Randomness (80-bit)
- Sortable: Lexicographic order = chronological order

**Format**: `01ARZ3NDEKTSV4RRFFQ69G5FAV`

### B. ISO 8601 Timestamp Format

```
YYYY-MM-DDTHH:mm:ss.sssZ
2025-12-12T14:30:00.000Z
```

### C. File Path Conventions

| Scope | Base Path |
|-------|-----------|
| Project AFS | `.context/` |
| Global AFS | `~/.context/` |
| App global | `~/.context/global/{app}/` |
| Shared global | `~/.context/global/shared/` |
| App config (hafs) | `~/.config/hafs/` |
| App config (oracle-code) | `~/.config/oracle-code/` |

### D. Cognitive Protocol Mapping

| AFS Directory | Cognitive Function | Access Policy |
|---------------|-------------------|---------------|
| `/memory` | Long-term memory, beliefs, values | read_only |
| `/knowledge` | Semantic knowledge, reference | read_only |
| `/history` | Episodic memory, transaction log | read_only |
| `/scratchpad` | Working memory, inner monologue | writable |
| `/tools` | Procedural memory, skills | executable |

### E. Research References

1. **AFS Paper**: "Everything is Context: Agentic File System Abstraction"
   - arXiv:2512.05470v1

2. **Mind Your Tone**: "Investigating How Prompt Politeness Affects LLM Accuracy"
   - arXiv:2510.04950v1

3. **Human-AI Synergy**: "Quantifying Human-AI Synergy"
   - Paper 7799

4. **Scaling Agents**: "Towards a Science of Scaling Agent Systems"
   - arXiv:2512.08296

5. **AutoCommenter**: "AI-Assisted Assessment of Coding Practices in Modern Code Review"
   - ACM 3664646.3665664

6. **ML Code Review**: "Resolving Code Review Comments with Machine Learning"
   - Paper 7525

7. **Where to Comment**: "Where should I comment my code?"
   - ACM 3377816.3381736

### F. Project Groups Example

```json
{
  "version": "1.0",
  "groups": {
    "zelda": {
      "description": "Zelda 3 ROM hacking projects",
      "projects": ["yaze", "oracle-of-secrets", "usdasm"],
      "shared_categories": ["knowledge", "decision", "preference"]
    },
    "agentic": {
      "description": "AFS and AI tooling projects",
      "projects": ["hafs", "oracle-code", "halext-code"],
      "shared_categories": ["knowledge", "decision", "preference"]
    },
    "halext-web": {
      "description": "Halext web infrastructure",
      "projects": ["halext-org", "halext-server"],
      "shared_categories": ["knowledge", "decision", "preference"]
    }
  },
  "project_paths": {
    "yaze": "~/Code/yaze",
    "oracle-of-secrets": "~/Code/Oracle-of-Secrets",
    "usdasm": "~/Code/usdasm",
    "hafs": "~/Code/hafs",
    "oracle-code": "~/Code/oracle-code",
    "halext-code": "~/Code/halext-code",
    "halext-org": "~/Code/halext-org",
    "halext-server": "~/Code/halext-server"
  }
}
```
