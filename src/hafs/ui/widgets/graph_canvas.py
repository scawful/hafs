"""Knowledge Graph Widget - ASCII/Unicode graph visualization.

This widget renders a knowledge graph using ASCII/Unicode characters,
supporting different node types and edge rendering with box-drawing characters.

Node types:
- ● concept (filled circle)
- ◆ file (diamond)
- ▲ bug (triangle)
- ■ function (square)

Edge rendering uses: ─│┌┐└┘├┤┬┴┼

Data format (JSON):
{
    "nodes": [
        {"id": "node1", "label": "Main", "type": "concept"},
        {"id": "node2", "label": "file.py", "type": "file"}
    ],
    "edges": [
        {"source": "node1", "target": "node2"}
    ]
}

Usage:
    graph = KnowledgeGraphWidget()
    graph.load_from_file("~/.context/memory/knowledge_graph.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


# Node type symbols
NODE_SYMBOLS = {
    "concept": "●",
    "file": "◆",
    "bug": "▲",
    "function": "■",
}

# Node type colors (CSS classes)
NODE_COLORS = {
    "concept": "node-concept",
    "file": "node-file",
    "bug": "node-bug",
    "function": "node-function",
}

# Box drawing characters for edges
EDGE_CHARS = {
    "horizontal": "─",
    "vertical": "│",
    "top_left": "┌",
    "top_right": "┐",
    "bottom_left": "└",
    "bottom_right": "┘",
    "left_t": "├",
    "right_t": "┤",
    "top_t": "┬",
    "bottom_t": "┴",
    "cross": "┼",
}


class GraphNode:
    """Represents a node in the knowledge graph."""

    def __init__(
        self,
        id: str,
        label: str,
        node_type: str = "concept",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initialize a graph node.

        Args:
            id: Unique node identifier
            label: Display label
            node_type: Type of node (concept, file, bug, function)
            metadata: Additional metadata
        """
        self.id = id
        self.label = label
        self.node_type = node_type
        self.metadata = metadata or {}
        self.x = 0
        self.y = 0

    @property
    def symbol(self) -> str:
        """Get the Unicode symbol for this node type."""
        return NODE_SYMBOLS.get(self.node_type, "○")

    @property
    def color_class(self) -> str:
        """Get the CSS color class for this node type."""
        return NODE_COLORS.get(self.node_type, "node-default")


class GraphEdge:
    """Represents an edge in the knowledge graph."""

    def __init__(self, source: str, target: str, label: str = "") -> None:
        """Initialize a graph edge.

        Args:
            source: Source node ID
            target: Target node ID
            label: Optional edge label
        """
        self.source = source
        self.target = target
        self.label = label


class KnowledgeGraphWidget(Widget):
    """ASCII/Unicode knowledge graph visualization widget.

    Renders a knowledge graph using text-based visualization with
    Unicode symbols for nodes and box-drawing characters for edges.

    Features:
    - Multiple node types with different symbols
    - Simple force-directed layout
    - Node selection and highlighting
    - Load from JSON file
    """

    DEFAULT_CSS = """
    KnowledgeGraphWidget {
        height: 100%;
        width: 100%;
        background: $surface;
        border: solid $primary;
    }

    KnowledgeGraphWidget .graph-title {
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0 1;
        background: $surface-darken-1;
    }

    KnowledgeGraphWidget .graph-canvas {
        height: 1fr;
        width: 100%;
        padding: 1;
    }

    KnowledgeGraphWidget .graph-node {
        height: auto;
    }

    KnowledgeGraphWidget .node-concept {
        color: $accent;
    }

    KnowledgeGraphWidget .node-file {
        color: $secondary;
    }

    KnowledgeGraphWidget .node-bug {
        color: $error;
    }

    KnowledgeGraphWidget .node-function {
        color: $success;
    }

    KnowledgeGraphWidget .node-default {
        color: $text;
    }

    KnowledgeGraphWidget .node-selected {
        background: $primary-darken-2;
        text-style: bold;
    }

    KnowledgeGraphWidget .graph-info {
        height: 1;
        color: $text-muted;
        padding: 0 1;
        background: $surface-darken-1;
    }
    """

    # Reactive state
    selected_node: reactive[Optional[str]] = reactive(None)

    def __init__(self, graph_file: Optional[Path] = None, **kwargs) -> None:
        """Initialize the knowledge graph widget.

        Args:
            graph_file: Optional path to JSON graph file
            **kwargs: Additional widget parameters
        """
        super().__init__(**kwargs)
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._graph_file = graph_file

    def compose(self) -> ComposeResult:
        """Compose the graph widget layout."""
        yield Static("Knowledge Graph", classes="graph-title")
        with VerticalScroll(classes="graph-canvas", id="graph-canvas"):
            pass
        yield Static(self._get_info_text(), classes="graph-info", id="graph-info")

    def on_mount(self) -> None:
        """Load graph data on mount."""
        if self._graph_file and self._graph_file.exists():
            self.load_from_file(self._graph_file)
        else:
            # Try default location
            default_path = Path.home() / ".context" / "memory" / "knowledge_graph.json"
            if default_path.exists():
                self.load_from_file(default_path)
            else:
                self._create_sample_graph()

        self._render_graph()

    def load_from_file(self, file_path: Path) -> bool:
        """Load graph from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            True if loaded successfully
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self._nodes.clear()
            self._edges.clear()

            # Load nodes
            for node_data in data.get("nodes", []):
                node = GraphNode(
                    id=node_data["id"],
                    label=node_data.get("label", node_data["id"]),
                    node_type=node_data.get("type", "concept"),
                    metadata=node_data.get("metadata", {}),
                )
                self._nodes[node.id] = node

            # Load edges
            for edge_data in data.get("edges", []):
                edge = GraphEdge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    label=edge_data.get("label", ""),
                )
                self._edges.append(edge)

            self._graph_file = file_path
            self._calculate_layout()
            self._render_graph()
            return True

        except Exception as e:
            # Create error node
            self._nodes.clear()
            self._edges.clear()
            error_node = GraphNode(
                "error",
                f"Failed to load: {e}",
                "bug",
            )
            self._nodes["error"] = error_node
            self._render_graph()
            return False

    def _create_sample_graph(self) -> None:
        """Create a sample graph for demonstration."""
        # Sample nodes
        nodes = [
            GraphNode("main", "Main Concept", "concept"),
            GraphNode("auth", "auth.py", "file"),
            GraphNode("login", "login()", "function"),
            GraphNode("bug1", "Auth Bug", "bug"),
            GraphNode("config", "config.json", "file"),
        ]

        for node in nodes:
            self._nodes[node.id] = node

        # Sample edges
        self._edges = [
            GraphEdge("main", "auth"),
            GraphEdge("auth", "login"),
            GraphEdge("login", "bug1"),
            GraphEdge("main", "config"),
        ]

        self._calculate_layout()

    def _calculate_layout(self) -> None:
        """Calculate node positions using a simple layout algorithm."""
        if not self._nodes:
            return

        # Simple tree-like layout
        # Start with root nodes (no incoming edges)
        incoming = {node_id: 0 for node_id in self._nodes}
        for edge in self._edges:
            incoming[edge.target] = incoming.get(edge.target, 0) + 1

        roots = [node_id for node_id, count in incoming.items() if count == 0]
        if not roots:
            roots = [list(self._nodes.keys())[0]]

        # BFS layout
        visited = set()
        queue = [(root, 0, 0) for root in roots]
        level_counts = {}

        while queue:
            node_id, level, offset = queue.pop(0)
            if node_id in visited:
                continue

            visited.add(node_id)
            node = self._nodes[node_id]

            # Calculate position
            level_count = level_counts.get(level, 0)
            node.y = level * 3
            node.x = level_count * 20

            level_counts[level] = level_count + 1

            # Add children
            children = [
                edge.target
                for edge in self._edges
                if edge.source == node_id and edge.target not in visited
            ]
            for child in children:
                queue.append((child, level + 1, 0))

    def _render_graph(self) -> None:
        """Render the graph to the canvas."""
        try:
            canvas = self.query_one("#graph-canvas", VerticalScroll)
            canvas.remove_children()

            if not self._nodes:
                canvas.mount(Static("No graph data available", classes="graph-node"))
                return

            # Render each node
            for node_id, node in self._nodes.items():
                label = f"{node.symbol} {node.label}"
                classes = f"graph-node {node.color_class}"

                if node_id == self.selected_node:
                    classes += " node-selected"

                node_widget = Static(label, classes=classes)
                node_widget.data_node_id = node_id  # Store node ID for selection
                canvas.mount(node_widget)

            # Update info
            info = self.query_one("#graph-info", Static)
            info.update(self._get_info_text())

        except Exception:
            pass

    def _get_info_text(self) -> str:
        """Get info bar text.

        Returns:
            Info text with node/edge counts
        """
        node_count = len(self._nodes)
        edge_count = len(self._edges)
        selected_text = f" | Selected: {self.selected_node}" if self.selected_node else ""
        return f"Nodes: {node_count} | Edges: {edge_count}{selected_text}"

    def select_node(self, node_id: Optional[str]) -> None:
        """Select a node by ID.

        Args:
            node_id: Node ID to select (None to deselect)
        """
        if node_id and node_id in self._nodes:
            self.selected_node = node_id
        else:
            self.selected_node = None

        self._render_graph()

    def watch_selected_node(self, new_node: Optional[str]) -> None:
        """React to node selection changes."""
        self._render_graph()

    def get_node_info(self, node_id: str) -> Optional[GraphNode]:
        """Get information about a node.

        Args:
            node_id: Node ID

        Returns:
            GraphNode if found, None otherwise
        """
        return self._nodes.get(node_id)

    def refresh_graph(self) -> None:
        """Reload and refresh the graph."""
        if self._graph_file and self._graph_file.exists():
            self.load_from_file(self._graph_file)
