import json
import os
import re
from typing import Dict, Any
from groq import Groq

# Reuse the StoryRequest dataclass
from story_mvp.generator import StoryRequest


class StoryLLM:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("STORY_MODEL", "llama-3.3-70b-versatile")
        
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY env var.")
        
        self.client = Groq(api_key=self.api_key)

    def generate(self, request: StoryRequest, rag_context: str) -> Dict[str, Any]:
        """Generates a story piece and metadata using ChatGroq based on the prompt."""
        
        system_prompt = self._build_system_prompt(request, rag_context)
        user_prompt = f"Idea: {request.idea}\nCharacters: {request.characters}"
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content
            return self._parse_json(content)
            
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            raise

    def _build_system_prompt(self, request: StoryRequest, rag_context: str) -> str:
        """Constructs the system prompt based on mode and injects RAG context."""
        
        mode_instructions = {
            "hook": "Write a compelling opening hook (2-3 sentences) for an audio story.",
            "expand": "Expand the provided scene idea into a vivid, detailed passage (2-4 paragraphs).",
            "continue": "Continue this story idea with natural progression and end on a cliffhanger (2-4 paragraphs)."
        }
        
        instruction = mode_instructions.get(request.mode, mode_instructions["continue"])
        
        prompt = f"""You are an expert audio-first storytelling AI writer.
{instruction}

Target Requirements:
- Genre: {request.genre}
- Tone: {request.tone}
- Language: {request.language} (If hindi or hinglish, seamlessly integrate it into dialogue or narration as appropriate for an Indian audience)
- Length preference: {request.length}

Context / Inspiration (Use these retrieved examples for style/tone reference, DO NOT copy them):
---
{rag_context}
---

You MUST respond in valid JSON format exactly matching this schema:
{{
  "output": "The actual generated story text. Use \\n\\n for paragraphs.",
  "pitch": "A 1-sentence catchy pitch for this story.",
  "story_bible": {{
    "central_conflict": "1 sentence describing the core conflict",
    "primary_characters": ["Char1", "Char2"],
    "recurring_image": "A specific sensory detail",
    "next_episode_question": "A cliffhanger question for the listener"
  }},
  "style_notes": [
    "Note on voice",
    "Note on pacing",
    "Note on language usage"
  ]
}}
"""
        return prompt

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Safely extracts and parses JSON from the LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON block if the model included markdown formatting
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
            raise ValueError(f"Could not parse valid JSON from LLM response: {text[:100]}...")
