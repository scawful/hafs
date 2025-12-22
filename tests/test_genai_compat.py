from __future__ import annotations

from dataclasses import dataclass

from core.genai_compat import (
    GenAIClient,
    extract_embeddings,
    extract_text,
    generate_content,
)


@dataclass
class DummyResponse:
    text: str = ""
    usage_metadata: object | None = None


class FakeAsyncModels:
    async def generate_content(self, model: str, contents: str) -> DummyResponse:
        return DummyResponse(text=f"{model}:{contents}")


class FakeAsyncClient:
    def __init__(self) -> None:
        self.models = FakeAsyncModels()


async def test_generate_content_prefers_async_client() -> None:
    client = GenAIClient(mode="genai", client=object(), async_client=FakeAsyncClient())
    response = await generate_content(client, "gemini-test", "hello")
    assert response.text == "gemini-test:hello"


def test_extract_text_from_candidates() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Hello "},
                        {"text": "world"},
                    ]
                }
            }
        ]
    }
    assert extract_text(response) == "Hello world"


def test_extract_embeddings_from_legacy_dict() -> None:
    response = {"embedding": [0.1, 0.2, 0.3]}
    assert extract_embeddings(response) == [0.1, 0.2, 0.3]
