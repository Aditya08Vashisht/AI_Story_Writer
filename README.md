# StoryTutor-MM MVP: Free-First Educational RAG

StoryTutor-MM is a multilingual study-story maker for Class 6–8 Science/Social Science. It preserves the original `/api/generate` contract while adding curriculum-grounded explanations, quizzes, storyboard plans, narration scripts, and video-demo plans.

## Free-First Stack

<<<<<<< HEAD
- **Default hosted model:** Groq Qwen via `GROQ_API_KEY` and `STORY_MODEL=qwen/qwen3-32b`.
- **Local fallback:** Ollama via `OLLAMA_MODEL=qwen3:1.7b` or another local model.
- **Optional Sarvam demo:** disabled by default. Enable only with `STORYTUTOR_ENABLE_SARVAM=1` plus `SARVAM_API_KEY`; use it for limited free-credit testing/evaluation, not as a required dependency.
- **Embeddings:** `STORYTUTOR_EMBEDDING_MODEL=BAAI/bge-m3` by default for multilingual retrieval.
- **Video path:** storyboard, narration, diagram specs, and local-renderable plans first; no paid image/video/TTS API required.
=======
<img width="1893" height="963" alt="image" src="https://github.com/user-attachments/assets/01e02f27-7e3b-4e8e-847c-114b844ba82d" />


## Features & Modes
>>>>>>> b82a28aca943eef7987006b5640d3e665c731144

## Curriculum Ingestion

PDFs live under `story_datasets/NCERT_6th-8th`. Convert them into processed JSON chunks:

```bash
python -m story_mvp.ingest_curriculum --source story_datasets/NCERT_6th-8th --output knowledge_base/processed
```

The ingester writes:

<<<<<<< HEAD
- `knowledge_base/processed/curriculum_chunks.json`
- `knowledge_base/processed/ingestion_audit.json`
=======
<img width="1892" height="950" alt="image" src="https://github.com/user-attachments/assets/6f5faf47-6471-4df1-8333-e787c7bd18d1" />


## Getting Started
>>>>>>> b82a28aca943eef7987006b5640d3e665c731144

Each chunk includes class, subject, language, chapter, page, source file, and text metadata. The audit flags missing Class 6–8 Science/Social Science coverage across English, Hindi, and Marathi.

## API

Existing payloads still work. New educational fields are additive:

```json
{
  "idea": "Explain photosynthesis using a story and video demo",
  "class_level": "6",
  "subject": "science",
  "language": "marathi",
  "chapter": "plants",
  "output_type": "video_demo_plan",
  "difficulty": "medium",
  "mode": "expand",
  "tone": "cinematic",
  "length": "medium"
}
```

Responses keep the original fields and add tutor artifacts:

```json
{
  "title": "...",
  "output": "...",
  "pitch": "...",
  "story_bible": {},
  "style_notes": [],
  "safety": {},
  "learning_objective": "...",
  "retrieved_sources": [],
  "explanation": "...",
  "story": "...",
  "quiz": [],
  "storyboard": [],
  "diagram_plan": {},
  "narration_script": "...",
  "video_demo_plan": {
    "scenes": [],
    "assets_needed": [],
    "renderable_without_paid_api": true
  }
}
```

## Architecture

1. **Curriculum ingestion:** PDF text is extracted into structured chunks.
2. **Hybrid retrieval:** FAISS vector search plus keyword scoring and metadata filters.
3. **Model adapter:** Groq Qwen first, optional Sarvam if explicitly enabled, Ollama fallback, deterministic fallback.
4. **Tutor pipeline:** responses include explanation, story, quiz, citations, storyboard, diagram plan, narration, and video-demo plan.
5. **Compatibility:** `/api/generate`, legacy payload keys, and original response keys remain stable.

## Environment

```env
GROQ_API_KEY=optional_for_groq_free_tier
STORY_MODEL=qwen/qwen3-32b
STORYTUTOR_EMBEDDING_MODEL=BAAI/bge-m3
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:1.7b

# Optional Sarvam testing only
STORYTUTOR_ENABLE_SARVAM=0
SARVAM_API_KEY=
SARVAM_MODEL=sarvam-30b
```

## Run

```bash
python -m story_mvp.app
```

Open:

```text
http://127.0.0.1:5050
```

## Test

```bash
python -m pytest tests/ -v
```
