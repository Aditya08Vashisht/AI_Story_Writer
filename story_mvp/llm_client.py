"""Compatibility LLM client for StoryTutor-MM.

The public class name stays `StoryLLM`, but internally it now uses a free-first
provider chain: Groq Qwen first, Ollama second.
"""

from __future__ import annotations

from typing import Any, Dict

from story_mvp.generator import StoryRequest
import os

from story_mvp.model_clients import (
    FallbackModelClient,
    GroqQwenClient,
    OllamaClient,
    SarvamClient,
    build_tutor_system_prompt,
    parse_json_response,
)


class StoryLLM:
    def __init__(self, api_key: str = None, model: str = None):
        clients = []
        try:
            clients.append(GroqQwenClient(api_key=api_key, model=model))
        except Exception:
            pass
        if os.environ.get("STORYTUTOR_ENABLE_SARVAM", "").lower() in {"1", "true", "yes"}:
            try:
                clients.append(SarvamClient())
            except Exception:
                pass
        clients.append(OllamaClient())
        self.client = FallbackModelClient(clients)
        self.last_provider = "none"
        self.last_error = ""

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        result = self.client.generate(request, rag_context)
        self.last_provider = self.client.last_provider
        self.last_error = self.client.last_error
        return result

    def _build_system_prompt(self, request: StoryRequest, rag_context: str) -> str:
        return build_tutor_system_prompt(request, rag_context)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return parse_json_response(text)
