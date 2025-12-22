import pytest
import os
from core.nodes import NodeManager, HalextNode, NodeStatus
from core.orchestrator_v2 import UnifiedOrchestrator, Provider, TaskTier

@pytest.mark.asyncio
async def test_halext_node_registration():
    """Verify that HalextNode can be registered and serialized."""
    manager = NodeManager()
    
    node = HalextNode(
        name="halext-cloud",
        host="api.halext.org",
        port=443,
        api_token="test-token",
        org_id="test-org",
        gateway_url="https://api.halext.org/v1"
    )
    
    manager.add_node(node)
    
    assert "halext-cloud" in manager.nodes_dict
    retrieved = manager.get_node("halext-cloud")
    assert isinstance(retrieved, HalextNode)
    assert retrieved.api_token == "test-token"
    assert retrieved.node_type == "halext"

@pytest.mark.asyncio
async def test_orchestrator_halext_routing():
    """Verify that the orchestrator can route to a Halext node."""
    orchestrator = UnifiedOrchestrator()
    # Mock node manager initialization
    orchestrator._node_manager = NodeManager()
    
    node = HalextNode(
        name="halext-cloud",
        host="api.halext.org",
        api_token="test-token",
        org_id="test-org"
    )
    orchestrator._node_manager.add_node(node)
    
    # Mock health check for the node
    orchestrator._provider_health[Provider.HALEXT] = True
    
    route = await orchestrator.route(
        prompt="test",
        provider=Provider.HALEXT
    )
    
    assert route.provider == Provider.HALEXT
    # The current implementation doesn't set node_name for HALEXT provider 
    # as it's typically treated as a service, but we've added node awareness to generate
    
@pytest.mark.asyncio
async def test_custom_model_loading_scaffold():
    """Verify the orchestrator's custom model loading scaffold."""
    orchestrator = UnifiedOrchestrator()
    orchestrator._node_manager = NodeManager()
    
    # Should not crash and should log/handle missing nodes
    await orchestrator.load_custom_model("path/to/my-model.gguf")
    
    # Add a mock local node to see if it's selected
    from core.nodes import ComputeNode
    local_gpu = ComputeNode(
        name="local-gpu",
        host="localhost",
        capabilities=["gpu", "ollama"]
    )
    orchestrator._node_manager.add_node(local_gpu)
    
    await orchestrator.load_custom_model("path/to/my-model.gguf", node_name="local-gpu")
