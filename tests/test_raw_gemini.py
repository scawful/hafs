import asyncio
import os

import pytest

genai = pytest.importorskip("google.genai")

async def test_raw_gemini():
    print("Testing Raw Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    
    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello"
        )
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_raw_gemini())
