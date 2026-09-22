"""Free-first model client adapters for StoryTutor-MM."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable

from groq import Groq

from story_mvp.generator import StoryRequest


class ModelClientError(RuntimeError):
    pass


class BaseModelClient:
    provider_name = "base"

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        raise NotImplementedError


class GroqQwenClient(BaseModelClient):
    provider_name = "groq"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("STORY_MODEL", "qwen/qwen3-32b")
        if not self.api_key:
            raise ModelClientError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=self.api_key)

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": build_tutor_system_prompt(request, rag_context)},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            model=self.model,
            temperature=0.6,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        return parse_json_response(completion.choices[0].message.content)


class OllamaClient(BaseModelClient):
    provider_name = "ollama"

    def __init__(self, model: str = None, host: str = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": build_tutor_system_prompt(request, rag_context)},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            "options": {"temperature": 0.6},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"Ollama request failed: {exc}") from exc
        message = body.get("message", {}).get("content", "")
        return parse_json_response(message)


class VLLMClient(BaseModelClient):
    """Self-hosted, OpenAI-compatible endpoint served by `vllm serve`.

    Runs on a Sol GPU node via slurm/serve_vllm.sbatch. vLLM's OpenAI-compatible
    server accepts the same `response_format` JSON-mode flag Groq uses, so this
    mirrors GroqQwenClient's request shape rather than Ollama's native /api/chat
    shape. No API key is required by a default vLLM server; it still expects an
    Authorization header, so a placeholder value is sent.
    """

    provider_name = "vllm"

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL", "")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.timeout = int(os.environ.get("VLLM_TIMEOUT", "120"))
        if not self.base_url:
            raise ModelClientError("VLLM_BASE_URL is not set.")
        if not self.model:
            raise ModelClientError("VLLM_MODEL is not set.")

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_tutor_system_prompt(request, rag_context)},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            "temperature": 0.6,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"vLLM request failed: {exc}") from exc
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_json_response(content)


class SarvamClient(BaseModelClient):
    provider_name = "sarvam"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model or os.environ.get("SARVAM_MODEL", "sarvam-30b")
        self.endpoint = os.environ.get("SARVAM_CHAT_ENDPOINT", "https://api.sarvam.ai/v1/chat/completions")
        if not self.api_key:
            raise ModelClientError("SARVAM_API_KEY is not set.")

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_tutor_system_prompt(request, rag_context)},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            "temperature": 0.6,
            "max_completion_tokens": 1800,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"Sarvam request failed: {exc}") from exc
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_json_response(content)


class FallbackModelClient(BaseModelClient):
    provider_name = "fallback_chain"

    def __init__(self, clients: Iterable[BaseModelClient]):
        self.clients = list(clients)
        self.last_provider = "none"
        self.last_error = ""

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        errors = []
        for client in self.clients:
            try:
                result = client.generate(request, rag_context)
                self.last_provider = client.provider_name
                self.last_error = ""
                return result
            except Exception as exc:
                errors.append(f"{client.provider_name}: {exc}")
        self.last_provider = "deterministic"
        self.last_error = "; ".join(errors)
        raise ModelClientError(self.last_error or "No model clients configured.")


def create_model_client() -> FallbackModelClient:
    clients = []
    try:
        clients.append(GroqQwenClient())
    except Exception:
        pass
    if os.environ.get("STORYTUTOR_ENABLE_SARVAM", "").lower() in {"1", "true", "yes"}:
        try:
            clients.append(SarvamClient())
        except Exception:
            pass
    try:
        clients.append(VLLMClient())
    except Exception:
        pass
    clients.append(OllamaClient())
    return FallbackModelClient(clients)


def build_user_prompt(request: StoryRequest) -> str:
    return "\n".join(
        [
            f"Learning goal: {request.idea}",
            f"Characters: {request.characters}",
            f"Class level: {request.class_level}",
            f"Subject: {request.subject}",
            f"Chapter/topic: {request.chapter or request.topic}",
            f"Output type: {request.output_type}",
            f"Difficulty: {request.difficulty}",
        ]
    )


def build_tutor_system_prompt(request: StoryRequest, rag_context: str) -> str:
    return f"""You are StoryTutor-MM, a free-first multilingual AI tutor for Class 6, Class 7, and Class 8 learners.
Teach Science and Social Science through curriculum-grounded explanations, memorable stories, misconceptions, quizzes, and video-demo planning.

Target:
- Language: {request.language}
- Class: {request.class_level or "unspecified"}
- Subject: {request.subject or "unspecified"}
- Tone: {request.tone}
- Length: {request.length}
- Output type: {request.output_type}

Use the retrieved sources as grounding. Do not invent textbook facts if the context is insufficient; say what needs verification.

Retrieved sources:
---
{rag_context}
---

Return only valid JSON with this schema:
{{
  "output": "Main student-facing response in the requested language.",
  "pitch": "One sentence learning summary.",
  "story_bible": {{
    "central_conflict": "The learning challenge or misconception.",
    "primary_characters": ["Tutor", "Learner"],
    "recurring_image": "Memorable phenomenon, map, diagram, or analogy.",
    "next_episode_question": "Reflective check-for-understanding question."
  }},
  "style_notes": ["Tutor voice note", "Pacing note", "Language note"],
  "learning_objective": "Specific curriculum goal.",
  "explanation": "Clear concept explanation.",
  "story": "Story-based learning version.",
  "quiz": [
    {{
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "answer": "Correct option text",
      "explanation": "Why this answer is correct"
    }}
  ],
  "storyboard": [
    {{
      "scene_number": 1,
      "visual": "What appears on screen",
      "narration": "Voiceover line",
      "learning_point": "Concept reinforced"
    }}
  ],
  "diagram_plan": {{
    "type": "flowchart",
    "title": "Diagram title",
    "nodes": ["Node 1", "Node 2"],
    "edges": [["Node 1", "Node 2"]]
  }},
  "narration_script": "Narration script for local TTS or human recording.",
  "video_demo_plan": {{
    "scenes": [],
    "assets_needed": [],
    "renderable_without_paid_api": true
  }}
}}
"""


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ModelClientError(f"Could not parse JSON response: {text[:120]}...")
