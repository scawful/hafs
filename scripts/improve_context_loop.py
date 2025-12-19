#!/usr/bin/env python3
import asyncio
import os
import sys
import re
import csv
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))
# Allow loading plugins if they are in PYTHONPATH (like hafs_google_internal)
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs_google_internal/src"))

from hafs.agents.swarm import SwarmCouncil
from hafs.core.quota import quota_manager
from hafs.core.plugin_loader import load_plugins, load_all_agents_from_package
from hafs.core.registry import agent_registry
from hafs.core.config import hafs_config, METRICS_DIR, KNOWLEDGE_DIR, CONTEXT_ROOT
import hafs.agents as agents_pkg

async def improve_context():
    print("--- Starting Context Improvement Loop ---")
    
    # 0. Load Plugins & Agents
    load_plugins()
    load_all_agents_from_package(agents_pkg)
    
    instantiated_agents = {}
    for name, cls in agent_registry.list_agents().items():
        try:
            instantiated_agents[name] = cls()
        except TypeError:
            pass # Skip agents needing args
    
    # 1. Check Health & Scale
    health = quota_manager.evaluate_system_health()
    scale = quota_manager.recommend_scale()
    
    if health == "CRITICAL":
        print("[ABORT] System in Quota Lockout. Exiting.")
        return

    # 2. Check Frequency
    metrics_dir = Path(METRICS_DIR).expanduser()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    last_run_file = metrics_dir / "last_context_run"
    
    last_run_time = 0.0
    if last_run_file.exists():
        try: last_run_time = float(last_run_file.read_text())
        except: pass
        
    now = time.time()
    elapsed = now - last_run_time
    
    # Check War Room Mode
    war_room_file = Path(CONTEXT_ROOT).expanduser() / "war_room_active"
    is_war_room = war_room_file.exists()

    # Thresholds (seconds)
    if is_war_room:
        print("🔥 WAR ROOM MODE ACTIVE 🔥")
        scale = "LOW" # Fast checks only
        thresholds = {"LOW": 30} # 30 seconds
    else:
        thresholds = {
            "HIGH": 300,    # 5 mins
            "MEDIUM": 3600, # 1 hour
            "LOW": 14400    # 4 hours
        }
    
    if elapsed < thresholds.get(scale, 300):
        print(f"[SKIP] Elapsed {elapsed/60:.1f}m < Threshold {thresholds.get(scale)/60:.1f}m for scale {scale}.")
        return

    # Update last run
    last_run_file.write_text(str(now))
    
    print(f"Running at scale: {scale} (War Room: {is_war_room})")
    council = SwarmCouncil(instantiated_agents)
    council.scale = scale
    await council.setup()
    
    if is_war_room:
        # In War Room, we focus on ONE thing: System Health
        print("[War Room] Focusing on Incident Signals...")
        await council.run_session("Incident Response")
        return

    # Load history to prevent duplication
    history_file = metrics_dir / "context_quality.csv"
    
    if not history_file.exists():
        history_file.write_text("timestamp,topic,score,notes\n")
        
    recent_topics = set()
    try:
        with open(history_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if within last 24h
                try:
                    ts = datetime.fromisoformat(row['timestamp'])
                    if (datetime.now() - ts).total_seconds() < 86400: # 24 hours
                        recent_topics.add(row['topic'])
                except: pass
    except Exception as e:
        print(f"Error reading history: {e}")

    print(f"Recent topics (skipping): {recent_topics}")

    # Refresh topics from trends
    trend_watcher = council.agents_map.get("TrendWatcher")
    discovered_topics = []
    if trend_watcher:
        try:
            print("Refreshing topic suggestions from TrendWatcher...")
            discovered_topics = await trend_watcher.run_task(ignore_topics=list(recent_topics))
        except Exception as e:
            print(f"Warning: TrendWatcher failed to propose topics: {e}")
    
    # Load topics from file and combine with discovered
    topics = []
    knowledge_dir = Path(KNOWLEDGE_DIR).expanduser()
    suggested_file = knowledge_dir / "suggested_topics.txt"
    if suggested_file.exists():
        topics = [t for t in suggested_file.read_text().splitlines() if t and t not in recent_topics]
    
    topics.extend([t for t in discovered_topics if t not in recent_topics and t not in topics])
        
    if not topics:
        print("No new topics found. Adding fallback 'Codebase Health'.")
        topics = ["Codebase Health"]

    # Loop through topics
    for i, topic in enumerate(topics[:3]): # Run top 3 unique
        print(f"\n[Iteration {i+1}] Running session on topic: {topic}")
        
        # Run session
        report = await council.run_session(topic)
        
        # Parse Score (Flexible match)
        match = re.search(r'(?:CONFIDENCE_SCORE:|Confidence Score)[:\*]*\s*(\d+)', report, re.IGNORECASE)
        score = int(match.group(1)) if match else 0
        
        print(f"  > Score: {score}")
        
        # Log
        with open(history_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), topic, score, f"Iteration {i+1}"])
            
        if score >= 85:
            print("[SUCCESS] Context Quality Target Met!")
        else:
            print("[ADJUSTING] Score too low. Refining search strategy...")
                
    print("\n--- Improvement Loop Complete ---")

if __name__ == "__main__":
    asyncio.run(improve_context())
