#!/usr/bin/env python3
"""Example demonstrating the HAFS multi-agent orchestration system.

This example shows how to:
1. Set up an AgentCoordinator with multiple agents
2. Route messages using @mentions and content-based routing
3. Use shared context for inter-agent communication
4. Broadcast messages to all agents
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from hafs.agents import AgentCoordinator, AgentRole


async def example_basic_setup():
    """Example 1: Basic coordinator setup and agent registration."""
    print("=" * 60)
    print("Example 1: Basic Setup")
    print("=" * 60)

    # Create coordinator with configuration
    config = {
        "max_agents": 5,
        "default_backend": "claude",
        "enable_context_sharing": True,
    }
    coordinator = AgentCoordinator(config)

    # Register multiple agents with different roles
    print("\nRegistering agents...")

    await coordinator.register_agent(
        name="alice",
        role=AgentRole.PLANNER,
        backend_name="claude",
        system_prompt="You are Alice, a strategic planning specialist.",
    )
    print("  ✓ Registered alice (PLANNER)")

    await coordinator.register_agent(
        name="bob",
        role=AgentRole.CODER,
        backend_name="claude",
        system_prompt="You are Bob, a coding implementation specialist.",
    )
    print("  ✓ Registered bob (CODER)")

    await coordinator.register_agent(
        name="carol",
        role=AgentRole.CRITIC,
        backend_name="claude",
        system_prompt="You are Carol, a code review and quality specialist.",
    )
    print("  ✓ Registered carol (CRITIC)")

    await coordinator.register_agent(
        name="dave",
        role=AgentRole.RESEARCHER,
        backend_name="claude",
        system_prompt="You are Dave, a research and investigation specialist.",
    )
    print("  ✓ Registered dave (RESEARCHER)")

    # Display registered agents
    print(f"\nTotal agents registered: {coordinator.agent_count}")
    print("Agents:", ", ".join(coordinator.list_agents()))

    return coordinator


async def example_message_routing(coordinator: AgentCoordinator):
    """Example 2: Message routing with @mentions and content-based inference."""
    print("\n" + "=" * 60)
    print("Example 2: Message Routing")
    print("=" * 60)

    # Example 1: Explicit routing with @mention
    print("\n1. Explicit routing with @mention:")
    message1 = "@alice create a project roadmap for building a web app"
    print(f"   Message: '{message1}'")
    recipient1 = await coordinator.route_message(message1, sender="user")
    print(f"   → Routed to: {recipient1}")

    # Example 2: Content-based routing (coder keywords)
    print("\n2. Content-based routing (coder keywords):")
    message2 = "implement the user authentication function"
    print(f"   Message: '{message2}'")
    recipient2 = await coordinator.route_message(message2, sender="user")
    print(f"   → Routed to: {recipient2} (inferred from 'implement')")

    # Example 3: Content-based routing (reviewer keywords)
    print("\n3. Content-based routing (reviewer keywords):")
    message3 = "review this code and check for security issues"
    print(f"   Message: '{message3}'")
    recipient3 = await coordinator.route_message(message3, sender="user")
    print(f"   → Routed to: {recipient3} (inferred from 'review' and 'check')")

    # Example 4: Content-based routing (researcher keywords)
    print("\n4. Content-based routing (researcher keywords):")
    message4 = "investigate the codebase and find all API endpoints"
    print(f"   Message: '{message4}'")
    recipient4 = await coordinator.route_message(message4, sender="user")
    print(f"   → Routed to: {recipient4} (inferred from 'investigate' and 'find')")

    # Example 5: Mention by role instead of name
    print("\n5. Mention by role:")
    message5 = "@planner what's the next step?"
    print(f"   Message: '{message5}'")
    recipient5 = await coordinator.route_message(message5, sender="user")
    print(f"   → Routed to: {recipient5} (matched PLANNER role)")


async def example_shared_context(coordinator: AgentCoordinator):
    """Example 3: Using shared context for coordination."""
    print("\n" + "=" * 60)
    print("Example 3: Shared Context")
    print("=" * 60)

    # Update shared context with project information
    print("\nUpdating shared context...")

    coordinator.update_shared_context(
        task="Build a REST API for a todo application"
    )
    print("  ✓ Set active task")

    # Add findings
    coordinator.update_shared_context(
        finding="Database schema already exists with users and tasks tables"
    )
    coordinator.update_shared_context(
        finding="Flask framework is installed and configured"
    )
    coordinator.update_shared_context(
        finding="JWT authentication library is available"
    )
    print("  ✓ Added 3 findings")

    # Add decisions
    coordinator.update_shared_context(
        decision="Use Flask-RESTful for API endpoints"
    )
    coordinator.update_shared_context(
        decision="Implement JWT-based authentication"
    )
    print("  ✓ Added 2 decisions")

    # Display the shared context
    print("\nShared context available to all agents:")
    print(coordinator.shared_context.to_prompt_text())

    # Now route a message - the context will be automatically injected
    print("Routing message with context injection...")
    message = "@bob implement the /api/tasks endpoint"
    print(f"Message: '{message}'")
    recipient = await coordinator.route_message(message, sender="user")
    print(f"→ Routed to: {recipient}")
    print("(Context automatically injected into bob's prompt)")


async def example_broadcast(coordinator: AgentCoordinator):
    """Example 4: Broadcasting messages to all agents."""
    print("\n" + "=" * 60)
    print("Example 4: Broadcasting")
    print("=" * 60)

    # Broadcast a message to all agents
    print("\nBroadcasting project update to all agents...")
    message = "Project deadline moved up by one week. Priority is now HIGH."
    recipients = await coordinator.broadcast(message, sender="system")

    print(f"Message broadcasted to {len(recipients)} agents:")
    for recipient in recipients:
        print(f"  ✓ {recipient}")


async def example_agent_status(coordinator: AgentCoordinator):
    """Example 5: Checking agent status."""
    print("\n" + "=" * 60)
    print("Example 5: Agent Status")
    print("=" * 60)

    # Get status of all agents
    print("\nAgent Status:")
    status = coordinator.get_all_status()

    for name, agent_status in status.items():
        print(f"\n{name}:")
        print(f"  Role: {agent_status['role']}")
        print(f"  Backend: {agent_status['backend']}")
        print(f"  Running: {agent_status['is_running']}")
        print(f"  Busy: {agent_status['is_busy']}")
        print(f"  Queue size: {agent_status['queue_size']}")


async def example_role_filtering(coordinator: AgentCoordinator):
    """Example 6: Filtering agents by role."""
    print("\n" + "=" * 60)
    print("Example 6: Role Filtering")
    print("=" * 60)

    print("\nAgents by role:")

    for role in AgentRole:
        agents = coordinator.list_agents_by_role(role)
        if agents:
            print(f"  {role.value}: {', '.join(agents)}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HAFS Multi-Agent Orchestration System Examples")
    print("=" * 60)

    # Example 1: Basic setup
    coordinator = await example_basic_setup()

    # Example 2: Message routing
    await example_message_routing(coordinator)

    # Example 3: Shared context
    await example_shared_context(coordinator)

    # Example 4: Broadcasting
    await example_broadcast(coordinator)

    # Example 5: Agent status
    await example_agent_status(coordinator)

    # Example 6: Role filtering
    await example_role_filtering(coordinator)

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up...")
    for agent_name in coordinator.list_agents():
        await coordinator.unregister_agent(agent_name)
    print("✓ All agents unregistered")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
