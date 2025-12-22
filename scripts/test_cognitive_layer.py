"""Test the Cognitive Layer functionality."""
import sys
import os
import json
from pathlib import Path

# Add source path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from core.cognitive import CognitiveLayer

def test_cognitive_layer():
    print("--- Testing Cognitive Layer ---")
    
    # 1. Initialization
    # Use a temp file for testing to avoid overwriting real state
    test_state_file = Path("/tmp/test_cognitive_state.json")
    if test_state_file.exists():
        test_state_file.unlink()
        
    cog = CognitiveLayer(state_file=test_state_file)
    print("✅ Initialized CognitiveLayer")

    # 2. Update State
    cog.update_state(emotional_state="CURIOSITY", confidence=0.85, thought="Testing the system.")
    print("✅ Updated state to CURIOSITY/0.85")
    
    # 3. Verify State in Memory
    if cog.state.emotional_state != "CURIOSITY":
        print(f"❌ State Mismatch: Expected CURIOSITY, got {cog.state.emotional_state}")
        return
    if cog.state.confidence != 0.85:
        print(f"❌ Confidence Mismatch: Expected 0.85, got {cog.state.confidence}")
        return
        
    # 4. Verify Context Injection string
    injection = cog.get_context_injection()
    print(f"Injection:\n{injection}")
    if "Emotional State: CURIOSITY" not in injection:
         print("❌ Injection string missing emotional state.")
         return
    print("✅ Context Injection Verified")

    # 5. Verify Persistence
    if not test_state_file.exists():
        print("❌ State file was not created.")
        return
    
    # Load raw JSON to check trace
    data = json.loads(test_state_file.read_text())
    trace = data.get("reasoning_trace", [])
    if len(trace) != 1:
         print(f"❌ Trace length mismatch. Expected 1, got {len(trace)}")
         return
         
    last_entry = trace[0]
    if last_entry["thought"] != "Testing the system.":
        print(f"❌ Trace thought mismatch: {last_entry['thought']}")
        return

    print("✅ Persistence & Trace Verified")
    
    # Cleanup
    test_state_file.unlink()

if __name__ == "__main__":
    test_cognitive_layer()
