import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import glob
import shutil
import difflib
import re

from core.registry import agent_registry
from core.config import hafs_config
from core.plugin_loader import load_plugins
# Need delegation for Command Center
# Assuming delegation manager is ported to core.delegation
try:
    from core.delegation import delegation_manager
except ImportError:
    delegation_manager = None

# --- UI Registry ---
class PageRegistry:
    def __init__(self):
        self.pages = {}
        
    def register_page(self, name, render_fn):
        self.pages[name] = render_fn

ui_registry = PageRegistry()

# --- 1. THEME & CONFIGURATION ---
st.set_page_config(
    page_title="HAFS Hub (Public)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic CSS (Safe Mode)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
CONTEXT_ROOT = hafs_config.context_root
KNOWLEDGE_DIR = CONTEXT_ROOT / "knowledge"
METRICS_DIR = CONTEXT_ROOT / "metrics"
LOGS_DIR = CONTEXT_ROOT / "background_agent" / "reports"
SCRATCHPAD_DIR = CONTEXT_ROOT / "scratchpad"
LOG_FILE = METRICS_DIR / "agents.jsonl"
BRIEFINGS_DIR = CONTEXT_ROOT / "background_agent" / "briefings"
KNOWLEDGE_GRAPH_FILE = CONTEXT_ROOT / "memory" / "knowledge_graph.json"
ASSETS_DIR = Path(__file__).parent / "assets"

# --- UTILS ---
def load_css():
    css_file = ASSETS_DIR / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

def render_heartbeat():
    """Renders the live activity feed from CognitiveLayer."""
    try:
        from core.cognitive import CognitiveLayer
        cog = CognitiveLayer()
        st.markdown("<h3>System Heartbeat</h3>", unsafe_allow_html=True)
        trace = cog.state.reasoning_trace
        if not trace:
            st.caption("No recent activity detected.")
            return

        heartbeat_html = '<div class="heartbeat-container">'
        for item in reversed(trace[-20:]):
            agent = item.get("agent", "Unknown")
            thought = item.get("thought", "")
            t_str = item.get("timestamp", "").split("T")[-1][:8]
            heartbeat_html += f"""
            <div class="heartbeat-item">
                <span class="heartbeat-time">[{t_str}]</span> 
                <span class="heartbeat-agent">{agent}</span><br/>
                <span class="heartbeat-thought">{thought}</span>
            </div>
            """
        heartbeat_html += '</div>'
        st.markdown(heartbeat_html, unsafe_allow_html=True)
    except:
        st.info("Cognitive Layer heartbeat unavailable.")

def get_file_tree(root_dir):
    file_list = []
    if not root_dir.exists(): return []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')] 
        dirs.sort()
        files.sort()
        rel_root = Path(root).relative_to(root_dir)
        depth = len(rel_root.parts)
        if rel_root == Path("."): depth = 0
        indent = "│  " * depth
        if depth > 0: file_list.append(f"{indent[:-3]}📁 **{rel_root.name}/**")
        for f in files:
            if f.startswith('.'): continue
            full_path = root_dir / rel_root / f
            file_list.append(f"{indent}📄 {f} ::{full_path}")
    return file_list

def parse_tree_selection(selection):
    if not selection or "::" not in selection: return None
    return Path(selection.split("::")[1])

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

@st.cache_data(ttl=60)
def discover_all_context_roots():
    scopes = {"Global": CONTEXT_ROOT}
    # Add workspace scopes if directory exists
    ws_dir = hafs_config.agent_workspaces_dir
    if ws_dir.exists():
        for d in ws_dir.iterdir():
            if d.is_dir() and (d / ".context").exists():
                scopes[f"WS: {d.name}"] = d / ".context"
    return scopes

# --- PAGES ---

def render_ops():
    st.markdown("<h1>Mission Control</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            from core.cognitive import CognitiveLayer
            cog = CognitiveLayer()
            m1, m2 = st.columns(2)
            m1.metric("System Confidence", f"{cog.state.confidence:.2f}")
            m2.metric("Emotional State", cog.state.emotional_state)
        except: pass
        render_heartbeat()
    with col2:
        st.markdown("<h3>Reports</h3>")
        report_files = sorted(list(LOGS_DIR.glob("*.md")), key=os.path.getmtime, reverse=True)
        if report_files:
            latest = report_files[0]
            with st.expander(f"Latest: {latest.name}", expanded=True):
                st.markdown(latest.read_text())
        else:
            st.caption("No reports yet.")

def render_intelligence_map():
    st.markdown("<h1>Intelligence Map</h1>", unsafe_allow_html=True)
    
    if not KNOWLEDGE_GRAPH_FILE.exists():
        st.warning(f"Knowledge Graph not yet built.")
        if st.button("🚀 Build Graph"):
            # This relies on the agent being registered
            st.info("Triggering build via CLI...")
            # For now, simplistic trigger
            return
        return
        
    with open(KNOWLEDGE_GRAPH_FILE, "r") as f:
        graph = json.load(f)
        
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    
    st.markdown(f"**Total Entities**: {len(nodes)} | **Total Relationships**: {len(edges)}")
    
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        agraph_nodes = []
        agraph_edges = []
        type_colors = {
            "concept": "#8b5cf6", "bug": "#ec4899", "cl": "#10b981", 
            "file": "#6366f1", "document": "#6366f1", "implicit": "#94a3b8"
        }
        for nid, n in nodes.items():
            ntype = str(n.get("type", "implicit")).lower()
            color = type_colors.get(ntype, "#94a3b8")
            agraph_nodes.append(Node(id=nid, label=nid[:20], title=str(n), color=color, size=20))
        for e in edges:
            agraph_edges.append(Edge(source=e["source"], target=e["target"], label=e.get("relation", "")))
        
        config = Config(width=None, height=800, directed=True, physics=True, solver='barnesHut',
                        barnesHut={"gravitationalConstant": -4000, "springConstant": 0.04, "springLength": 250, "damping": 0.09})
        agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    except ImportError:
        st.error("Install streamlit-agraph to view the interactive graph.")


def render_knowledge_explorer():
    st.markdown("<h1>Knowledge Explorer</h1>", unsafe_allow_html=True)
    tree = get_file_tree(KNOWLEDGE_DIR)
    c_nav, c_main = st.columns([1, 3])
    with c_nav:
        selection = st.radio("Files", tree, label_visibility="collapsed")
        path = parse_tree_selection(selection)
    with c_main:
        if path and path.exists():
            st.markdown(f"## {path.name}")
            st.markdown(path.read_text())

def render_command_center():
    st.markdown("<h1>Agent Command Center</h1>", unsafe_allow_html=True)
    st.info("Directly instruct agents and override autonomous decisions.")
    
    agent_options = ["auto"] + list(agent_registry.agents.keys())
    selected_agent = st.selectbox("Direct Task To:", agent_options)
    task_input = st.text_area("Specific Instruction", placeholder="e.g. Analyze the current build failure...")
    
    if st.button("Dispatch Agent"):
        if task_input and delegation_manager:
            with st.spinner(f"Dispatched to {selected_agent}..."):
                async def run_delegation():
                    return await delegation_manager.delegate("agent", selected_agent, task_input)
                
                result = asyncio.run(run_delegation())
                st.success("Task dispatched.")
                st.markdown(f"**Result:** {result}")
        elif not delegation_manager:
            st.error("Delegation Manager not available.")
        else:
            st.warning("Please provide a task instruction.")

def render_failure_hub():
    st.markdown("<h1>Failure Hub</h1>", unsafe_allow_html=True)
    st.markdown("### Test Failure Analyst")
    inv_id = st.text_input("Sponge Invocation ID")
    if st.button("Analyze Failure") and inv_id:
        st.info("Analysis logic requires plugin support (SpongeAdapter).")

def render_agents():
    st.markdown("<h1>Agents</h1>", unsafe_allow_html=True)
    agents = agent_registry.list_agents()
    for name, cls in agents.items():
        with st.expander(f"🤖 {name}"):
            st.markdown(f"**Role:** {cls.__doc__ or 'N/A'}")

def render_workbench():
    st.markdown("<h1>Workbench</h1>", unsafe_allow_html=True)
    tools_dir = CONTEXT_ROOT / "tools"
    if tools_dir.exists():
        tools = [f.name for f in tools_dir.glob("*") if f.is_file()]
        sel = st.selectbox("Tool", tools)
        args = st.text_input("Args")
        if st.button("Run"):
            st.info(f"Running {sel} {args}...")

def render_infrastructure():
    st.markdown("<h1>Infrastructure</h1>", unsafe_allow_html=True)

    st.markdown("### Nodes")
    try:
        from core.nodes import node_manager
        run_async(node_manager.load_config())
        run_async(node_manager.health_check_all())
        node_rows = []
        for node in node_manager.nodes:
            node_rows.append({
                "name": node.name,
                "status": node.status.value,
                "host": node.host,
                "port": node.port,
                "type": node.node_type,
                "platform": node.platform,
                "latency_ms": node.latency_ms,
                "capabilities": ", ".join(node.capabilities),
                "models": ", ".join(node.models),
            })
        if node_rows:
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True)
        else:
            st.caption("No nodes configured.")
    except Exception as exc:
        st.error(f"Failed to load nodes: {exc}")

    st.markdown("### AFS Sync Status")
    sync_status_file = METRICS_DIR / "afs_sync_status.json"
    if not sync_status_file.exists():
        st.caption("No sync status recorded yet.")
        return

    try:
        data = json.loads(sync_status_file.read_text())
    except Exception as exc:
        st.error(f"Failed to load sync status: {exc}")
        return

    profiles = data.get("profiles", {})
    rows = []
    for profile_name, profile_data in profiles.items():
        targets = profile_data.get("targets", {})
        for target_name, record in targets.items():
            rows.append({
                "profile": profile_name,
                "target": target_name,
                "direction": record.get("direction"),
                "ok": record.get("ok"),
                "exit_code": record.get("exit_code"),
                "last_seen": record.get("timestamp"),
                "duration_ms": record.get("duration_ms"),
                "dry_run": record.get("dry_run"),
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No sync entries recorded.")

def main():
    # Load Plugins (to register UI pages)
    load_plugins()
    
    # Register core pages
    ui_registry.register_page("Operations", render_ops)
    ui_registry.register_page("Intelligence Map", render_intelligence_map)
    ui_registry.register_page("Command Center", render_command_center)
    ui_registry.register_page("Knowledge", render_knowledge_explorer)
    ui_registry.register_page("Failure Hub", render_failure_hub)
    ui_registry.register_page("Agents", render_agents)
    ui_registry.register_page("Workbench", render_workbench)
    ui_registry.register_page("Infrastructure", render_infrastructure)

    load_css()
    with st.sidebar:
        st.markdown("## HAFS Hub")
        # Dynamic Navigation
        page_names = list(ui_registry.pages.keys())
        # Default to first page if available
        index = 0
        page = st.radio("Navigation", page_names, index=index)
        
    # Render selected page
    if page in ui_registry.pages:
        ui_registry.pages[page]()

if __name__ == "__main__":
    main()
