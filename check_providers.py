import asyncio
import os
from core.orchestrator_v2 import UnifiedOrchestrator, Provider

async def check():
    print("Checking Providers...")
    u = UnifiedOrchestrator()
    await u.initialize()
    status = u.get_provider_status()
    for prov, info in status.items():
        print(f"Provider {prov}: {'AVAILABLE' if info['available'] else 'NOT AVAILABLE'}")
    
    print("\nEnvironment Variables:")
    for var in ["GEMINI_API_KEY", "AISTUDIO_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(var)
        print(f"{var}: {'SET' if val else 'NOT SET'}")

if __name__ == "__main__":
    asyncio.run(check())
