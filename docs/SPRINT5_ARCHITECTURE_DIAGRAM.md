# Sprint 5: Chat Mode Architecture Diagram

## Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     HAFS TUI - Chat Mode                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Orchestration (Existing)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────────┐          ┌──────────────────┐           │
│   │ AgentCoordinator │◄─────────┤   AgentLane      │           │
│   │                  │          │   (per agent)    │           │
│   └──────────────────┘          └──────────────────┘           │
│            │                              │                      │
│            │ (business logic)             │ (agent state)       │
│            ▼                              ▼                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Adapter (NEW - Sprint 5)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│              ┌────────────────────────────┐                      │
│              │     ChatAdapter            │                      │
│              │  - send_user_message()     │                      │
│              │  - stream_agent_response() │                      │
│              │  - execute_tool()          │                      │
│              │  - publish_agent_status()  │                      │
│              └────────────────────────────┘                      │
│                         │                                         │
│                         │ (translation)                          │
│                         ▼                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Protocol (NEW - Sprint 5)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│   │ ChatMessage  │    │StreamingCtx  │    │ MessageRole  │    │
│   │ - user()     │    │ - append()   │    │ - USER       │    │
│   │ - assistant()│    │ - complete() │    │ - ASSISTANT  │    │
│   │ - system()   │    │ - duration() │    │ - SYSTEM     │    │
│   │ - to_event() │    └──────────────┘    │ - TOOL       │    │
│   └──────────────┘                         └──────────────┘    │
│                                                                   │
│   Helper Functions:                                              │
│   • emit_chat_message()     • emit_stream_token()               │
│   • emit_stream_complete()  • emit_tool_result()                │
│                         │                                         │
│                         │ (events)                               │
│                         ▼                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Event Bus (Existing Core)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│              ┌────────────────────────────┐                      │
│              │       EventBus             │                      │
│              │  - publish()               │                      │
│              │  - subscribe()             │                      │
│              │  - pattern matching        │                      │
│              └────────────────────────────┘                      │
│                         │                                         │
│        ┌────────────────┼────────────────┐                      │
│        │                │                │                      │
│        ▼                ▼                ▼                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │ChatEvent │    │StreamTok │    │ToolEvent │                 │
│  └──────────┘    └──────────┘    └──────────┘                 │
│                         │                                         │
│                         │ (pub/sub)                              │
│                         ▼                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: UI Widgets (NEW - Sprint 5)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────────────────────────────────────────────┐     │
│   │         StreamingMessage Widget                       │     │
│   │  ┌────────────────────────────────────────────────┐  │     │
│   │  │ Header: ● Planner streaming...                 │  │     │
│   │  ├────────────────────────────────────────────────┤  │     │
│   │  │ Content:                                       │  │     │
│   │  │ • Subscribes to "chat.stream_token"            │  │     │
│   │  │ • Appends tokens (<50ms latency)               │  │     │
│   │  │ • Renders markdown via Rich                    │  │     │
│   │  │ • Auto-scrolls to newest                       │  │     │
│   │  └────────────────────────────────────────────────┘  │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                   │
│   ┌──────────────────────────────────────────────────────┐     │
│   │            ToolCard Widget                            │     │
│   │  ┌────────────────────────────────────────────────┐  │     │
│   │  │ Header: ✓ pytest (1.52s) [▼ Expand] [Copy]   │  │     │
│   │  ├────────────────────────────────────────────────┤  │     │
│   │  │ Content (collapsed):                           │  │     │
│   │  │ • Subscribes to "tool.result"                  │  │     │
│   │  │ • Shows summary when collapsed                 │  │     │
│   │  │ • Syntax highlights when expanded              │  │     │
│   │  │ • Copy button with feedback                    │  │     │
│   │  └────────────────────────────────────────────────┘  │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 6: Screen Integration (Existing + Updates)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│              ┌────────────────────────────┐                      │
│              │      ChatScreen            │                      │
│              │  - Mounts widgets          │                      │
│              │  - Subscribes to events    │                      │
│              │  - Uses ChatAdapter        │                      │
│              └────────────────────────────┘                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Event Flow Example: User sends "Hello"

```
1. User Input
   └─> ChatScreen.send_message("Hello")

2. Adapter Layer
   └─> adapter.send_user_message("Hello")
       ├─> emit_chat_message(bus, ChatMessage.user("Hello"))
       │   └─> EventBus.publish(ChatEvent)
       │       └─> UI updates (user message bubble appears)
       │
       └─> coordinator.route_message("Hello")
           └─> Returns target agent: "planner"

3. Response Streaming
   └─> adapter.stream_agent_response("planner")
       ├─> Creates StreamingContext(message_id="abc123")
       │
       ├─> async for token in coordinator.stream():
       │   ├─> emit_stream_token(bus, "I", "abc123", "planner")
       │   ├─> emit_stream_token(bus, " ", "abc123", "planner")
       │   ├─> emit_stream_token(bus, "understand", "abc123", "planner")
       │   └─> ... (continues)
       │
       └─> emit_stream_complete(bus, "abc123", "planner", full_text)
           ├─> emit_stream_token(..., is_final=true)
           └─> emit_chat_message(ChatMessage.assistant(...))

4. Widget Updates
   ├─> StreamingMessage receives StreamTokenEvent
   │   ├─> Filters by message_id and agent_id
   │   ├─> Appends token to buffer (<10ms)
   │   ├─> Re-renders markdown
   │   └─> Auto-scrolls
   │
   └─> On is_final=true:
       └─> Marks streaming complete (● → ✓)
```

## Tool Execution Flow Example

```
1. Agent Executes Tool
   └─> adapter.execute_tool("coder", "pytest", {path: "tests/"})

2. Status Update
   └─> emit AgentStatusEvent(agent_id="coder", status="executing")

3. Tool Execution
   └─> coordinator.execute_tool(...)
       ├─> Runs pytest
       ├─> Captures stdout/stderr
       └─> Times execution (1523ms)

4. Result Publishing
   └─> emit_tool_result(bus, ToolResultEvent(
           tool_name="pytest",
           stdout="All tests passed\\n...",
           stderr="",
           duration_ms=1523,
           success=True,
           agent_id="coder"
       ))

5. Widget Creation
   └─> ChatScreen._on_tool_result(event)
       ├─> card = ToolCard.from_event(event)
       ├─> container.mount(card)
       └─> Card displays:
           ┌────────────────────────────────────┐
           │ ✓ pytest (1.52s) [▼ Expand] [Copy]│
           ├────────────────────────────────────┤
           │ All tests passed...                │
           └────────────────────────────────────┘
```

## Message State Lifecycle

```
ChatMessage Lifecycle:

PENDING
  │
  ├─> User sends message
  │   └─> Status: COMPLETE (instant)
  │
  └─> Agent generates response
      │
      ├─> Status: STREAMING
      │   ├─> Tokens arrive: "I" "am" "thinking"...
      │   └─> StreamingMessage appends each token
      │
      └─> Status: COMPLETE
          └─> Final message event published
```

## Component Dependencies

```
StreamingMessage
  ├─ Requires: EventBus (subscribe)
  ├─ Uses: RichLog (Textual)
  └─ Imports: chat_protocol, event_bus

ToolCard
  ├─ Requires: EventBus (subscribe)
  ├─ Uses: RichLog, Button (Textual)
  └─ Imports: event_bus

ChatAdapter
  ├─ Requires: AgentCoordinator
  ├─ Requires: EventBus (publish)
  └─ Imports: chat_protocol, event_bus

chat_protocol
  ├─ Requires: EventBus (publish)
  └─ Imports: event_bus (events)
```

## Performance Hot Paths

### Token Streaming (Critical Path)
```
Token arrives (0ms)
  ↓
emit_stream_token() (0.1ms)
  ↓
EventBus.publish() (0.5ms)
  ↓
Pattern matching (0.2ms)
  ↓
StreamingMessage._on_stream_token() (0.5ms)
  ↓
append_token() - filter check (0.1ms)
  ↓
Buffer append (0.1ms)
  ↓
RichLog.write() (5-8ms)
  ↓
Auto-scroll (1ms)
  ↓
Total: ~8-10ms (well under 50ms target)
```

### Tool Result Display
```
Tool completes (0ms)
  ↓
emit_tool_result() (0.1ms)
  ↓
EventBus.publish() (0.5ms)
  ↓
ChatScreen._on_tool_result() (1ms)
  ↓
ToolCard.from_event() (0.5ms)
  ↓
container.mount() (10-15ms)
  ↓
Total: ~15-20ms (smooth)
```

## Extension Points

### Future Enhancements
1. **Message Threading**: Add `parent_id` to ChatMessage
2. **Reactions**: Add `reactions: List[str]` to ChatMessage
3. **Edit History**: Add `edited_at: datetime` to ChatMessage
4. **Voice Notes**: Add `audio_url: str` to ChatMessage metadata
5. **File Attachments**: Add `attachments: List[Path]` to ChatMessage
6. **Message Search**: Subscribe to all ChatEvents, index content
7. **Export**: Collect ChatEvents, format as markdown/JSON

### Plugin Architecture
```python
class ChatPlugin:
    def on_message(self, msg: ChatMessage) -> None:
        """Called for every message."""
        pass

    def on_stream_start(self, ctx: StreamingContext) -> None:
        """Called when streaming starts."""
        pass

    def on_tool_result(self, result: ToolResultEvent) -> None:
        """Called for tool results."""
        pass

# Register plugin
adapter.register_plugin(LoggingPlugin())
adapter.register_plugin(PersistencePlugin())
adapter.register_plugin(AnalyticsPlugin())
```

This architecture enables clean separation of concerns and makes the chat system highly extensible while maintaining performance.
