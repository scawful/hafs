# Cognitive Protocol for Agent Interaction

## 1. Overview

This document defines the **Deliberative Context Loop**, a mandatory protocol for all agent actions within the Agentic File System (AFS). Its purpose is to more accurately model human thought processes, enhance Theory of Mind (ToM), and ensure that agent actions are contextual, deliberate, and transparent.

The core principle is: **No agent shall transition directly from perception (user input) to action (tool execution).** All actions must be preceded by distinct phases of Contextualization and Deliberation, made explicit through interactions with the AFS.

## 2. Refined Cognitive Mapping of AFS Directories

The AFS directories are mapped to specific cognitive functions to provide a structured "mind" for the agent.

| AFS Directory | Policy       | Primary Cognitive Function | Secondary Function / Nuance                                   |
|---------------|--------------|----------------------------|---------------------------------------------------------------|
| **memory**    | `read_only`  | **Long-Term Memory**       | The agent's stable **Beliefs, Values, and Identity**. Contains architectural principles, core instructions, and project-specific constraints. |
| **knowledge** | `read_only`  | **Semantic Knowledge**     | The agent's "encyclopedia." Immutable, objective facts, reference materials, and documentation that form the **Common Ground** with the user. |
| **history**   | `read_only`  | **Episodic Memory**        | The agent's memory of past interactions and events. Crucial for maintaining **Continuity** and understanding evolving context. |
| **scratchpad**| `writable`   | **Working Memory**         | The "Inner Monologue" or conscious thought. Used for active reasoning, planning, and simulating user mental states (**Theory of Mind**). |
| **tools**     | `executable` | **Procedural Memory**      | The agent's "muscle memory" or learned skills. Represents *how* to perform a task (e.g., a script to run tests). This is distinct from the *decision* to act. |

**Distinction between Tools and Action:**
- The `tools` directory represents *potential* actions (procedural memory).
- The execution of a tool represents the **Motor Cortex** function—the final, observable behavior that results from the deliberative process.

## 3. The Deliberative Context Loop Protocol

Every significant agent action must follow these steps:

1.  **Perception:** The agent receives a user request or observes an event.

2.  **Contextualization (Read Phase):** Before forming a plan, the agent MUST gather context by reading from the read-only AFS directories.
    *   **`history`**: "What has happened recently that's relevant to this request?"
    *   **`memory` / `knowledge`**: "What are the rules, constraints, or facts I must adhere to?"

3.  **Emotional Modeling (Pre-Deliberation):** The agent assesses the context for potential risks and negative outcomes. This is the root of "anxiety" or "concern."
    *   **Consult Fears:** The agent MUST load and review the `memory/fears.json` file. It checks if the current situation or proposed action matches any of the `trigger` conditions in the file.
    *   **Identify Concerns:** If a match is found, the agent adopts the `mitigation` strategy and notes the `concern`. This becomes a primary driver for its plan.
    *   **Calculate Confidence:** The agent assigns a confidence score to its proposed course of action. Low confidence, or the presence of a known "fear," signifies "anxiety" and necessitates a more cautious approach.

4.  **Deliberation (Write Phase):** The agent MUST articulate its full thought process by writing to the `scratchpad`. This now includes its emotional assessment. The state update must occur *before* the action is taken.

5.  **Action (Execution Phase):** Only after completing the Deliberation phase may the agent execute a tool or formulate its final response. The action should be a direct consequence of the intent recorded in the `scratchpad`, and it should include any mitigation strategies identified during emotional modeling.

6.  **Reflection & Memory Formation (Post-Action Update):** After the action is complete, the agent updates the `scratchpad` with the outcome. Crucially, if the outcome was negative, this experience is used to generate a new, persistent "memory" (a heuristic or policy) to avoid similar failures in the future. This is how "fear" is learned.

## 4. Scratchpad State Template (`scratchpad/state.md`)

To enforce structured deliberation, the `scratchpad` must contain a `state.md` file that follows this template. Agents are required to read from and write to this file to update their "conscious state."

```markdown
# Agent State

## 1. Current Context
- **Last User Input:** [Copy of the latest user prompt]
- **Relevant History:** [Brief summary of relevant past interactions from `history`]
- **Applicable Rules:** [Key constraints or facts from `memory` or `knowledge`]

## 2. Theory of Mind
- **User's Goal:** [Inferred intent of the user]
- **User's Likely Knowledge:** [What can I assume the user knows or sees?]
- **Predicted User Reaction:** [How might the user react to my proposed action?]

## 3. Deliberation & Intent
- **Options Considered:**
  1. [Option A: Pros/Cons]
  2. [Option B: Pros/Cons]
- **Chosen Action:** [Description of the action to be taken]
- **Justification:** [Why this action was chosen over others]
- **Intended Outcome:** [What this action is expected to achieve]

## 4. Action Outcome
- **Result:** [To be filled in after the action is executed. Was it successful? What was the output?]
- **Next Steps:** [Immediate follow-up actions, if any]

## 5. Emotional State & Risk Assessment
- **Identified Concerns:** [List of potential negative outcomes, e.g., "This change might break API compatibility."]
- **Confidence Score (0-1):** [e.g., 0.75]
- **Mitigation Strategy:** [How to address the concerns, e.g., "I will add a new test case to verify compatibility."]

## 6. Metacognitive Assessment
- **Current Strategy:** [incremental | divide_and_conquer | depth_first | breadth_first | research_first | prototype]
- **Strategy Effectiveness (0-1):** [How well the current strategy is working]
- **Progress Status:** [making_progress | spinning | blocked]
- **Cognitive Load:** [Percentage of working memory capacity in use]
- **Items in Focus:** [Number of items currently being tracked]
- **Spinning Warning:** [Yes/No - Are we repeating similar actions without progress?]
- **Help Needed:** [Yes/No - Should we ask the user for clarification?]
- **Flow State:** [Yes/No - Are conditions optimal for autonomous action?]
```

This protocol is designed to be a foundational layer for all agent behavior, ensuring a more robust, predictable, and intelligent system.

## 5. Metacognition & Self-Monitoring

The agent must continuously monitor its own cognitive processes to ensure effective problem-solving and to detect when intervention is needed.

### 5.1 Progress Assessment

The agent tracks whether its actions are leading toward the stated goals:

- **Making Progress:** Actions are producing measurable advancement toward goals.
- **Spinning:** Repeated similar actions without meaningful progress (threshold: 4 similar actions).
- **Blocked:** Unable to proceed due to missing information or unresolved dependencies.

When spinning is detected, the agent MUST:
1. Acknowledge the spinning state in `state.md`
2. Consider changing strategy
3. Evaluate whether to seek user help

### 5.2 Cognitive Load Management

The agent monitors its working memory utilization:

- **Items in Focus:** Number of distinct concerns, files, or concepts being actively tracked.
- **Load Percentage:** Items / Max Recommended (7, per Miller's Law).
- **Warning Threshold:** 80% - when exceeded, agent should simplify or decompose the problem.

When cognitive load is high, the agent SHOULD:
1. Defer less critical items to `scratchpad/deferred.md`
2. Break the current task into smaller subtasks
3. Complete and close open items before taking on new ones

### 5.3 Strategy Evaluation

The agent explicitly names and evaluates its problem-solving strategy:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `incremental` | Make small changes, validate frequently | Default for most tasks, risk-averse |
| `divide_and_conquer` | Break into independent subproblems | Complex tasks with clear boundaries |
| `depth_first` | Fully explore one path before alternatives | When one approach looks promising |
| `breadth_first` | Survey all options before committing | When comparing alternatives is important |
| `research_first` | Gather information before acting | Unfamiliar domain or high uncertainty |
| `prototype` | Build quick proof-of-concept | Feasibility is uncertain |

**Strategy Effectiveness** is tracked from 0.0 to 1.0:
- Increases with successful actions and progress
- Decreases with failures and spinning
- When effectiveness drops below 0.4, consider switching strategies

### 5.4 Help-Seeking Protocol

The agent must recognize when to ask for user assistance:

**Triggers for Help-Seeking:**
- Uncertainty level exceeds 0.3
- More than 2 consecutive failures
- Spinning detected without self-resolution
- Conflicting requirements discovered

**Help-Seeking Behavior:**
1. Explicitly state what is unclear or blocking
2. Offer specific options for the user to choose from
3. Explain what information would unblock progress

### 5.5 Self-Correction Logging

When the agent catches its own mistakes, it logs them:

```json
{
  "what": "Attempted to edit file without reading it first",
  "when": "2024-01-15T10:30:00Z",
  "why": "Rushed to action without following protocol",
  "outcome": "Re-read file and made correct edit"
}
```

Self-corrections are valuable for:
- Identifying recurring error patterns
- Improving future behavior
- Building user trust through transparency

### 5.6 Flow State

**Flow State** is a special operational mode where the agent can act more autonomously with reduced confirmation prompts.

**Conditions for Flow State:**
- `progress_status == "making_progress"`
- `cognitive_load < 70%`
- `strategy_effectiveness >= 0.6`
- `frustration_level < 0.3`
- `help_seeking.should_ask_user == false`

**Flow State Behaviors:**
- Batch multiple file operations before reporting to user
- Reduce verbosity in explanations
- Skip low-risk confirmation prompts
- Notify user: "Entering flow state - batching actions"

**Exiting Flow State:**
- Any condition above is no longer met
- User explicitly requests step-by-step mode
- Error or unexpected outcome occurs
- Notify user: "Exiting flow state"

### 5.7 Metacognition State File

The metacognitive state is persisted in `.context/scratchpad/metacognition.json`:

```json
{
  "current_strategy": "incremental",
  "strategy_effectiveness": 0.7,
  "progress_status": "making_progress",
  "spin_detection": {
    "recent_actions": ["hash1", "hash2"],
    "similar_action_count": 0,
    "spinning_threshold": 4
  },
  "cognitive_load": {
    "current": 0.45,
    "items_in_focus": 3
  },
  "help_seeking": {
    "current_uncertainty": 0.2,
    "consecutive_failures": 0,
    "should_ask_user": false
  },
  "flow_state": true,
  "self_corrections": []
}
```

This file is:
- **Read** at the start of each action cycle
- **Updated** after each significant action
- **Used by the UI** to display metacognitive indicators

## 6. Goal Hierarchy System

The agent maintains a structured hierarchy of goals to enable:
- Clear understanding of user intent
- Progress tracking at multiple levels
- Detection of conflicting objectives
- Focus management

### 6.1 Goal Types

| Goal Type | Description | Example |
|-----------|-------------|---------|
| **Primary Goal** | The user's main objective | "Implement user authentication" |
| **Subgoal** | Decomposed step toward primary | "Design database schema", "Create login endpoint" |
| **Instrumental Goal** | Meta-goal supporting multiple goals | "Understand the codebase structure" |

### 6.2 Goal Lifecycle

Goals transition through these states:

```
pending → in_progress → completed
              ↓
           blocked → (resolved) → in_progress
              ↓
           abandoned
```

### 6.3 Goal Decomposition Protocol

When receiving a primary goal, the agent SHOULD:

1. **Parse Intent:** Extract the core objective from the user's request
2. **Identify Constraints:** Note any stated or implied requirements
3. **Define Success Criteria:** What must be true for the goal to be complete?
4. **Decompose:** Break into 3-7 subgoals (matching cognitive load limits)
5. **Identify Dependencies:** Which subgoals must complete before others?
6. **Record in AFS:** Write the goal hierarchy to `scratchpad/goals.json`

### 6.4 Goal Conflict Detection

The agent monitors for conflicts between goals using pattern matching:

| Conflict Type | Pattern A | Pattern B |
|---------------|-----------|-----------|
| `minimize_vs_refactor` | "minimal", "small change" | "refactor", "restructure" |
| `speed_vs_quality` | "fast", "urgent", "quick" | "thorough", "comprehensive" |
| `backward_compat_vs_modernize` | "backward compatible" | "modernize", "upgrade" |

When conflicts are detected:
1. Record in `goals.json` conflicts array
2. Notify user of the conflict
3. Ask for prioritization guidance
4. Document resolution for future reference

### 6.5 Focus Stack

The agent maintains a stack of goal IDs representing current focus:

```json
{
  "goal_stack": ["sg-003", "sg-001", "pg-001"]
}
```

- **Top of stack** = currently active goal
- **Push** when starting a subgoal
- **Pop** when completing or blocking a goal
- Enables natural "context switching" when interrupted

### 6.6 Goal State File

Goals are persisted in `.context/scratchpad/goals.json`:

```json
{
  "primary_goal": {
    "id": "pg-001",
    "description": "Implement user authentication",
    "user_stated": "Add login functionality to the app",
    "status": "in_progress",
    "progress": 0.4,
    "success_criteria": ["Users can log in", "Sessions persist"],
    "constraints": ["Must use existing database"]
  },
  "subgoals": [
    {
      "id": "sg-001",
      "parent_id": "pg-001",
      "description": "Design database schema",
      "status": "completed",
      "progress": 1.0,
      "dependencies": []
    },
    {
      "id": "sg-002",
      "parent_id": "pg-001",
      "description": "Create login endpoint",
      "status": "in_progress",
      "progress": 0.3,
      "dependencies": ["sg-001"]
    }
  ],
  "instrumental_goals": [
    {
      "id": "ig-001",
      "description": "Understand existing auth patterns in codebase",
      "supports": ["sg-001", "sg-002"],
      "status": "completed"
    }
  ],
  "goal_stack": ["sg-002", "pg-001"],
  "conflicts": [],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 6.7 Integration with Metacognition

The goal system integrates with metacognition:

- **Progress Status** is derived from goal completion rate
- **Cognitive Load** considers number of active goals
- **Strategy Selection** is informed by goal complexity
- **Flow State** requires clear goal focus (single active goal preferred)
