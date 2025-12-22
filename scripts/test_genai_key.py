"""Stress test for GenAI and API Key verification."""
import asyncio
import os
import sys
import google.generativeai as genai

# Load Config
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))
from core.config import hafs_config

async def test_key():
    print("--- GenAI Auth Stress Test ---")
    
    # 1. Get Key
    api_key = hafs_config.aistudio_api_key
    if not api_key:
        print("❌ FAIL: API Key not found in config.")
        return
        
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"Using API Key: {masked_key}")
    
    # 2. Configure SDK
    genai.configure(api_key=api_key)
    
    # 3. Test: List Models (Auth Check)
    print("\nAttempting to list models...")
    try:
        models = genai.list_models()
        count = sum(1 for _ in models)
        print(f"✅ Auth Success! Found {count} available models.")
    except Exception as e:
        print(f"❌ Auth Failed: {e}")
        return

    # 4. Test: Multiple Generations
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    for i in range(3):
        print(f"\n[Request {i+1}] Generating...")
        try:
            # We use synchronous call for simplicity in this loop
            # as it's a dedicated test script.
            response = model.generate_content("Say 'Auth verified' and then a random fact.")
            print(f"Response: {response.text.strip()}")
            if response.usage_metadata:
                print(f"Usage: {response.usage_metadata.total_token_count} tokens.")
        except Exception as e:
            print(f"❌ Request {i+1} Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_key())
