# StoryTutor-MM MVP

StoryTutor-MM is a demo-ready AI tutor interface for story-based educational content. It preserves the original API shape while shifting the product from story creation to curriculum-grounded tutoring.

- Generate a learning hook
- Expand a concept explanation
- Continue an adaptive learning story

The backend can use a RAG + LLM pipeline when configured, with a local deterministic fallback so the demo still works without external services.

## Run

```bash
python -m story_mvp.app
```

Open:

```text
http://127.0.0.1:5050
```

If Flask is not installed yet, open this file directly in a browser:

```text
story_mvp/demo.html
```

Or serve the static demo with Node:

```bash
node story_mvp/serve_static.js
```

Open:

```text
http://127.0.0.1:5051
```

## API

```http
POST /api/generate
```

Example payload:

```json
{
  "mode": "expand",
  "genre": "thriller",
  "tone": "suspenseful",
  "language": "hinglish",
  "length": "medium",
  "characters": "Rhea, Kabir",
  "idea": "Help a student understand energy transfer through a story-based explanation."
}
```

## Future Architecture

Keep the response fields stable while evolving internals toward hybrid RAG, LangGraph agents, learner memory, MCP servers, multimodal generation, and assessment:

- `output`
- `pitch`
- `story_bible`
- `style_notes`
- `safety`
