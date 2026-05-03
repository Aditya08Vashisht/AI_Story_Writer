# Story Studio MVP

Story Studio is a demo-ready AI co-writer interface for serialized audio storytelling. It focuses on three MVP tasks:

- Generate a hook
- Expand a scene
- Continue a story

The current backend is a local deterministic story engine, so the demo works without downloading model weights. The API is shaped so it can later be swapped for a fine-tuned LLM service.

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
  "idea": "A podcast writer receives voice notes from a missing listener."
}
```

## Future Model Swap

Replace `story_mvp/generator.py` with an adapter that calls a fine-tuned model endpoint. Keep the response fields stable:

- `output`
- `pitch`
- `story_bible`
- `style_notes`
- `safety`
