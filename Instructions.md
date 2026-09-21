# Instructions — StoryTutor-MM MVP

Operational guide for setting up, running, ingesting, testing, and debugging this
project. Everything here is derived from the code as it currently stands.

---

## 1. What you are running

A Flask web app that turns a learner's "learning goal" into a curriculum-grounded
study artifact (explanation + story + quiz + storyboard + narration + video plan),
retrieved from ingested NCERT Class 6–8 Science / Social Science textbooks in
English, Hindi, and Marathi.

Entry point: [story_mvp/app.py](story_mvp/app.py) → `http://127.0.0.1:5050`

---

## 2. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python | 3.10+ (verified working on 3.14.2) |
| Disk | ~3.5 GB — the repo already carries 282 NCERT PDFs plus a 25 MB processed chunk file |
| RAM | 4 GB+ free. The default embedding model `BAAI/bge-m3` is large (~2 GB download) |
| Network | Needed once to download the embedding model, and per-request if using Groq |
| Optional | [Ollama](https://ollama.com) for a fully local LLM fallback |
| Optional | Node.js, only for the static demo server [story_mvp/serve_static.js](story_mvp/serve_static.js) |

---

## 3. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the six direct dependencies:
`flask`, `groq`, `sentence-transformers`, `faiss-cpu`, `python-dotenv`, `pypdf`.
`torch` and `numpy` arrive transitively through `sentence-transformers`.

---

## 4. Configure `.env`

Create `.env` in the project root. It is already listed in `.gitignore` — never
commit it, and rotate any key that has been shared.

Only these variables are actually read by the code:

| Variable | Read in | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | [model_clients.py](story_mvp/model_clients.py) | — | Enables the Groq provider. Without it, Groq is silently skipped |
| `STORY_MODEL` | [model_clients.py](story_mvp/model_clients.py) | `qwen/qwen3-32b` | Groq model id |
| `STORYTUTOR_EMBEDDING_MODEL` | [rag_engine.py](story_mvp/rag_engine.py) | `BAAI/bge-m3` | Sentence-transformers model used for the FAISS index |
| `OLLAMA_HOST` | [model_clients.py](story_mvp/model_clients.py) | `http://127.0.0.1:11434` | Local LLM endpoint |
| `OLLAMA_MODEL` | [model_clients.py](story_mvp/model_clients.py) | `qwen3:1.7b` | Local LLM model tag |
| `STORYTUTOR_ENABLE_SARVAM` | [llm_client.py](story_mvp/llm_client.py) | unset (off) | Set to `1`/`true`/`yes` to add Sarvam to the provider chain |
| `SARVAM_API_KEY` | [model_clients.py](story_mvp/model_clients.py) | — | Required only when Sarvam is enabled |
| `SARVAM_MODEL` | [model_clients.py](story_mvp/model_clients.py) | `sarvam-30b` | Sarvam model id |
| `SARVAM_CHAT_ENDPOINT` | [model_clients.py](story_mvp/model_clients.py) | `https://api.sarvam.ai/v1/chat/completions` | Sarvam endpoint override |

A minimal working file:

```env
GROQ_API_KEY=your_groq_key
STORY_MODEL=llama-3.3-70b-versatile
STORYTUTOR_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> **Known dead config:** `RAG_TOP_K` appears in the current `.env` but nothing
> reads it. `top_k` defaults to `3` in
> [rag_engine.py:106](story_mvp/rag_engine.py#L106) and the generator never
> overrides it. Change the default in code, or wire the env var up, if you need
> a different value.

---

## 5. First run — read this before you start the server

[app.py](story_mvp/app.py) builds the RAG engine **at import time**, synchronously,
before Flask starts serving. On a cold start with the default corpus that means:

1. Load 7,992 curriculum chunks (25 MB JSON) from `knowledge_base/`.
2. Re-serialize all of them to compute a SHA-256 dataset signature.
3. Download and load the embedding model.
4. Embed all 7,992 chunks and write a FAISS `IndexFlatIP`.

With `BAAI/bge-m3` on CPU this is a long, silent wait (tens of minutes). Two ways
to make the first run tolerable:

- **Fast path for development:** set `STORYTUTOR_EMBEDDING_MODEL=all-MiniLM-L6-v2`.
  It is a much smaller model and is what the test suite uses.
- **Accept the cost once:** the index is cached under `story_mvp/faiss_index/`
  and reused as long as the document count *and* the content signature match.

The cache filename is derived from the model name, e.g. `baai_bge_m3.index`
plus a matching `.signature` file. A leftover `story_rag.index` from an earlier
version of the code is inert — the current code never reads it and it can be
deleted.

If initialization fails for any reason, the app does **not** crash. It prints
`Warning: Failed to initialize AI components` and serves the deterministic
template engine instead.

---

## 6. Run

### Flask app (full pipeline)

```bash
python -m story_mvp.app
```

Open `http://127.0.0.1:5050`.

`run_story_studio()` deliberately passes `use_reloader=False` so debug mode does
not initialize the heavy RAG engine twice.

### Static demo (no Python, no backend)

Open [story_mvp/demo.html](story_mvp/demo.html) directly in a browser, or serve it:

```bash
node story_mvp/serve_static.js      # http://127.0.0.1:5051
```

[story.js](story_mvp/static/story.js) detects the missing backend, flips the
status pill to `Static Demo`, and generates output entirely client-side from its
built-in `GENRE_DATA` templates.

---

## 7. Curriculum ingestion

Re-run this whenever you add or replace PDFs under `story_datasets/`.

```bash
python -m story_mvp.ingest_curriculum \
  --source story_datasets/NCERT_6th-8th \
  --output knowledge_base/processed
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--source` | `story_datasets/NCERT_6th-8th` | Root folder to walk for `.pdf` files |
| `--output` | `knowledge_base/processed` | Where chunks and audit are written |
| `--chunk-chars` | `1800` | Target characters per chunk |
| `--overlap-chars` | `250` | Overlap between consecutive chunks |
| `--limit` | `0` (all) | Process only the first N PDFs — use for smoke tests |
| `--checkpoint-every` | `10` | Write partial output every N PDFs so a crash is not fatal |
| `--workers` | `min(cpu_count, 4)` | Parallel extraction processes |

Outputs:

- `knowledge_base/processed/curriculum_chunks.json` — the indexable corpus
- `knowledge_base/processed/ingestion_audit.json` — PDF count, per-bucket
  coverage, extraction errors, and `missing_expected_coverage`

Current state of the corpus: **282 PDFs → 7,992 chunks, 0 errors, 0 missing
coverage buckets** across all 18 combinations of class (6/7/8) × subject
(science/social_science) × language (english/hindi/marathi).

Metadata is inferred from the **file path**, not the PDF contents — see
`_metadata_from_path()` in
[ingest_curriculum.py:115](story_mvp/ingest_curriculum.py#L115). If you add PDFs,
the folder name must contain the class (`Class6`, `Class7th`, …), the subject
(`science` / `sst` / `social`), and the language. Otherwise the chunk lands in an
`unknown` bucket and metadata filters will not find it.

> Writes are atomic: each JSON goes to `<name>.tmp` and is then `os.replace`d.
> A killed run leaves the last checkpoint intact rather than a truncated file.

---

## 8. API reference

### `GET /api/health`

```json
{
  "status": "online",
  "engine": "free_first_hybrid_rag",
  "embedding_model": "BAAI/bge-m3",
  "model_provider": "groq",
  "ready_for_model_swap": true,
  "timestamp": "2026-08-18T12:00:00"
}
```

`engine` reads `local_story_tutor_mvp` and `model_provider` reads
`deterministic` when AI initialization failed — this is the quickest way to tell
which engine actually served you.

### `GET /api/options`

Returns the valid enum values for every dropdown: modes, genres, tones,
languages, lengths, class levels, subjects, output types, difficulties.

### `GET /api/demo`

Returns the three canned demo payloads used by the "Load Demo" button.

### `POST /api/generate`

Request — every field is optional and independently validated:

```json
{
  "mode": "expand",
  "idea": "Explain photosynthesis using a story and video demo",
  "class_level": "6",
  "subject": "science",
  "language": "marathi",
  "chapter": "plants",
  "topic": "plants",
  "output_type": "video_demo_plan",
  "difficulty": "medium",
  "genre": "thriller",
  "tone": "cinematic",
  "length": "medium",
  "characters": "Aarav, Meera"
}
```

Any value outside the allowed set is replaced by a safe default rather than
rejected — `mode` → `continue`, `genre` → `thriller`, `tone` → `cinematic`,
`language` → `english`, `length` → `medium`. An empty `idea` gets a generic
placeholder goal. See `_normalize_request()` in
[generator.py:191](story_mvp/generator.py#L191).

Response — legacy fields first, then the tutor additions:

| Field | Type | Notes |
| --- | --- | --- |
| `title`, `mode`, `mode_label`, `genre`, `tone`, `language` | string | Echo / derived |
| `output` | string | Main learner-facing text |
| `pitch` | string | One-line summary |
| `story_bible` | object | `central_conflict`, `primary_characters`, `recurring_image`, `next_episode_question` |
| `style_notes` | string[] | Voice / pacing / format |
| `safety` | object | `status` (`clear`/`review`), `flags`, `note` |
| `learning_objective` | string | |
| `explanation` | string | |
| `story` | string | |
| `quiz` | object[] | `question`, `options[4]`, `answer`, `explanation` |
| `storyboard` | object[] | `scene_number`, `visual`, `narration`, `learning_point` |
| `diagram_plan` | object | `type`, `title`, `nodes`, `edges` |
| `narration_script` | string | |
| `video_demo_plan` | object | `scenes`, `assets_needed`, `renderable_without_paid_api`, `recommended_free_tools` |
| `retrieved_sources` | object[] | Citation metadata per retrieved chunk |
| `model_provider` | string | `groq` / `sarvam` / `ollama` / `deterministic` |
| `free_first` | bool | Always `true` |
| `generated_at` | string | ISO timestamp, added in the route |

The key-set is identical whether the AI path or the deterministic path served
the request — [test_ai_integration.py](tests/test_ai_integration.py) asserts this
explicitly.

---

## 9. Tests

```bash
python -m pytest tests/ -v
```

| File | Needs network / keys? | What it covers |
| --- | --- | --- |
| [test_story_mvp.py](tests/test_story_mvp.py) | No | Deterministic generator, defaults, educational fields |
| [test_storytutor_options.py](tests/test_storytutor_options.py) | No | Request normalization |
| [test_curriculum_ingestion.py](tests/test_curriculum_ingestion.py) | No | Path→metadata inference, coverage gap detection |
| [test_rag_engine.py](tests/test_rag_engine.py) | Downloads embedding model | Index build, filters, citations, language handling |
| [test_llm_client.py](tests/test_llm_client.py) | **Yes — skips without `GROQ_API_KEY`** | Live LLM JSON contract |
| [test_ai_integration.py](tests/test_ai_integration.py) | **Yes — skips without `GROQ_API_KEY`** | Full RAG+LLM pipeline, response-shape parity |

Offline subset (fast, no downloads, currently 10 passing):

```bash
python -m pytest tests/test_story_mvp.py tests/test_storytutor_options.py tests/test_curriculum_ingestion.py -q
```

Both RAG-touching test modules set `STORYTUTOR_EMBEDDING_MODEL=all-MiniLM-L6-v2`
and build into their own temp index directories, then delete them — running
tests never disturbs `story_mvp/faiss_index/`.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Warning: Failed to initialize AI components` at startup | Missing package, or embedding model could not load | Check the printed exception; confirm `pip install -r requirements.txt` completed |
| Server appears to hang on start | Cold FAISS build over 7,992 chunks | Wait, or switch to `all-MiniLM-L6-v2`, or reduce the corpus |
| `AI generation failed, falling back` per request | Every model provider in the chain failed | Check `GROQ_API_KEY`, model id validity, and whether Ollama is running |
| Output is obviously templated prose about locked phones and floorboards | The deterministic engine served the request | Check `/api/health` → `model_provider` |
| `retrieved_sources` is always empty | Deterministic path, or every retrieval filter excluded all docs | Loosen `chapter`/`topic` in the request |
| Index rebuilds on every start | Chunk file changed, or count mismatch | Expected — the signature check is content-based |
| `pypdf is required` | Optional import failed during ingestion | `pip install pypdf` |
| Marathi/Hindi request returns English sources | Deliberate — `RetrievalService` back-fills with English when native results are thin | Flagged per document as `retrieval_language_fallback` |

---

## 11. Operational notes and cautions

- `app.config["SECRET_KEY"]` is a hardcoded literal in
  [app.py:38](story_mvp/app.py#L38). Move it to an environment variable before
  any deployment where sessions matter.
- The Flask dev server is not a production server. Put Gunicorn/Waitress in
  front of it, and pre-build the FAISS index rather than building on boot.
- `_safety_check()` is a three-term substring match and self-documents as a demo
  filter. Do not treat it as moderation.
- `_dataset_signature()` re-serializes the entire 25 MB corpus on every startup.
  Fine for an MVP; the first thing to optimize if boot time matters.
- The 282 PDFs are committed to the repository. That is why a clone is ~3.5 GB.
