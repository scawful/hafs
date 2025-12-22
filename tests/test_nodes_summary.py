from __future__ import annotations

from core.nodes import ComputeNode, NodeManager, NodeStatus


def test_summary_includes_error_message() -> None:
    manager = NodeManager()
    node = ComputeNode(name="test-node", host="example.com")
    node.status = NodeStatus.OFFLINE
    node.error_message = "Timeout"
    manager._nodes[node.name] = node

    summary = manager.summary()

    assert "Timeout" in summary
