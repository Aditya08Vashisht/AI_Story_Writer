# Master Prompt — Run StoryTutor-MM on the GPU Machine

The project has already been copied to the GPU machine in full and opened in
VS Code. Paste everything below the divider into Claude Code / Cursor / any
coding agent running in that workspace.

One note for the sender: the transfer included `.env`, so the `GROQ_API_KEY` in
it now exists on a second machine. Rotate it. The GPU setup below does not need
it — the local model becomes the primary provider.

---

You are setting up an existing Python project, **StoryTutor-MM**, on this GPU
machine. It was developed on a CPU-only Windows laptop where it is impractically
slow. The full working tree has already been copied here and is open in VS Code.

Your job: clean out the artifacts that came along from the Windows machine, get
the GPU actually doing the work, run it, and verify it end to end.

## 1. What this project is

A Flask web app that turns a learner's "learning goal" into a curriculum-grounded
study bundle: explanation, story, quiz, storyboard, diagram plan, narration
script, video-demo plan, and citations. Aimed at Indian middle-school learners
(Class 6–8 Science and Social Science) in English, Hindi, and Marathi.

The grounding corpus is 282 NCERT textbook PDFs already ingested into **7,992
text chunks**, indexed with FAISS over sentence-transformer embeddings and passed
to an LLM as retrieval context.

Read `concept.md`, `Instructions.md`, and `skills.md` in the project root before
you start. They are current and accurate — architecture, API contract, and the
deliberate trade-offs. This prompt assumes you have read them.

### Module map

| File | Role |
| --- | --- |
| `story_mvp/app.py` | Flask routes; builds the RAG engine at import time |
| `story_mvp/generator.py` | Request normalization, orchestration, deterministic fallback engine, response assembly |
| `story_mvp/rag_engine.py` | `StoryRAG` (embeddings + FAISS) and `RetrievalService` (educational subclass) |
| `story_mvp/model_clients.py` | Groq / Ollama / Sarvam adapters, prompt construction, JSON parsing |
| `story_mvp/llm_client.py` | Thin `StoryLLM` shim over the provider chain |
| `story_mvp/ingest_curriculum.py` | Offline PDF → chunk pipeline (CPU/IO-bound) |
| `story_mvp/static/story.js` | Vanilla-JS frontend with an offline fallback generator |

## 2. Delete these first

All of it is Windows-machine residue or stale cache. Nothing here is recoverable
work, and nothing else in the project references any of it.

```bash
rm -rf .git
rm -rf story_mvp/__pycache__ tests/__pycache__ .pytest_cache
rm -rf story_mvp/faiss_index
rm -f  knowledge_base/processed/ingest-*.log
rm -rf models
rm -f  .env          # recreate it in section 4
```

Why each:

- **`.git/` (976 MB) — delete it, and this one matters.** The git history is an
  older, broken snapshot: `model_clients.py`, `ingest_curriculum.py`, and three
  test files were never committed, and the committed `llm_client.py` predates the
  provider-chain rewrite. Running `git checkout .`, `git stash`, or `git restore`
  in this workspace would silently revert working code to versions that fail on
  import. Removing `.git` makes that impossible. Re-init a fresh repo later if
  you want version control.
- **`__pycache__/`, `.pytest_cache/`** — CPython 3.14 Windows bytecode. Useless
  here and a source of confusing import behaviour.
- **`story_mvp/faiss_index/`** — contains one orphaned `story_rag.index` from an
  older naming scheme that the current code never reads. The index must be
  rebuilt on this machine regardless.
- **`knowledge_base/processed/ingest-*.log`** — stale run logs.
- **`models/`** — empty directory, nothing references it.
- **`.env`** — points at the old machine's Groq setup. You will write a new one.

**Do not delete `knowledge_base/`** — `processed/curriculum_chunks.json` (25 MB)
is the entire corpus and is not in git anywhere. Losing it means re-ingesting all
282 PDFs.

**Keep `story_datasets/NCERT_6th-8th/` (2.5 GB)** unless disk is tight. It is
only needed to re-run ingestion, but it is the only copy of the source PDFs.

## 3. Verify the GPU before changing anything

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Report the GPU model and VRAM back to me. The project shipped with `torch+cpu`
installed, so if `torch.cuda.is_available()` is `False` you will need to
reinstall torch with a CUDA build matching this machine's driver — use the
official selector at pytorch.org, do not guess a version.

## 4. Install and configure

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then reinstall torch with the correct CUDA wheel for this box
```

Six direct dependencies: `flask`, `groq`, `sentence-transformers`, `faiss-cpu`,
`python-dotenv`, `pypdf`. Everything else is transitive.

**Keep `faiss-cpu`. Do not install `faiss-gpu`.** The index is a flat
inner-product index over ~8,000 vectors — search is sub-millisecond on CPU. GPU
FAISS buys nothing and adds a fragile dependency. The GPU belongs on embeddings
and on the local LLM. Revisit only above roughly a million vectors.

### The GPU changes the model strategy

The provider chain in `StoryLLM.__init__` is Groq → (Sarvam, opt-in) → Ollama.
`GroqQwenClient` raises during construction when `GROQ_API_KEY` is absent and is
silently skipped. So **simply omitting `GROQ_API_KEY` makes the local Ollama
model the primary provider** — no code change needed. On a GPU box that is the
right call: fully local, free, and private, which is the project's stated
"free-first" principle taken to its conclusion.

Create a new `.env`:

```env
# Embeddings — GPU makes the good multilingual model practical
STORYTUTOR_EMBEDDING_MODEL=BAAI/bge-m3
STORYTUTOR_DEVICE=cuda
STORYTUTOR_EMBED_BATCH=64

# Local LLM as primary provider (deliberately no GROQ_API_KEY line)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:14b-instruct
OLLAMA_TIMEOUT=180
```

`STORYTUTOR_DEVICE`, `STORYTUTOR_EMBED_BATCH`, and `OLLAMA_TIMEOUT` do not exist
yet — you add them in section 5.

Install Ollama and pull a model sized to the VRAM you reported. Choose something
in the 7B–32B range that reliably emits strict JSON; the pipeline requests
JSON-formatted output and a weak model will fail to parse. Larger is better as
long as it fits comfortably.

## 5. Code changes — exactly these four

**5a. Explicit device selection —** `story_mvp/rag_engine.py` line 31

`self.model = SentenceTransformer(self.model_name)` becomes an explicit,
overridable device choice: read `STORYTUTOR_DEVICE` from the environment, fall
back to `"cuda"` when `torch.cuda.is_available()` and `"cpu"` otherwise, pass it
as `device=`. Sentence-transformers auto-detects CUDA, but being explicit lets us
force CPU for debugging and lets us *log* the choice — right now there is no way
to tell from the outside which device was used. Print the resolved device next to
the existing `print("Loading embedding model...")`.

**5b. Batched encoding with progress —** `story_mvp/rag_engine.py` line 76

Add `batch_size` (from `STORYTUTOR_EMBED_BATCH`, default 64),
`show_progress_bar=True`, and `normalize_embeddings=True` to the `.encode()`
call. Leave the existing `faiss.normalize_L2(embeddings)` on the next line — it
is a no-op on already-unit vectors, and removing it would be a silent
correctness risk if that flag ever changes.

Without a batch size the default is small and the GPU idles between batches.
Without the progress bar a multi-minute build looks like a hang.

**5c. Ollama timeout —** `story_mvp/model_clients.py` line 78

`urlopen(req, timeout=30)` was tuned for a tiny 1.7B model. A 14B model
generating the full ~1,800-token JSON bundle will exceed 30s, every request will
fall back to the deterministic engine, and the app will *look* like it is
working. Read the timeout from `OLLAMA_TIMEOUT`, default 180.

This is the single most likely cause of "it runs but the output is bad."

**5d. Change nothing else.** Specifically:

- The `/api/generate` request and response contracts are frozen. Adding or
  renaming a response field breaks `tests/test_ai_integration.py`, which asserts
  the AI and deterministic engines return identical key sets.
- Leave the deterministic fallback engine in place. It is why the app never
  hard-fails.
- Leave `escapeHtml()` in `story_mvp/static/story.js` — model output is untrusted
  input and that is the XSS guard.

## 6. Build the index

The FAISS index is built at Flask import time, synchronously, before the server
accepts connections. On CPU with `bge-m3` this took tens of minutes. On GPU it
should be minutes. Build it once, deliberately, before starting the server:

```bash
python -c "
import time, os
os.environ.setdefault('STORYTUTOR_DEVICE','cuda')
from story_mvp.rag_engine import RetrievalService
t=time.time()
r=RetrievalService(dataset_dir='knowledge_base')
print('device:', r.model.device)
print('documents:', len(r.documents))
print('vectors:', r.index.ntotal, 'dim:', r.index.d)
print('seconds:', round(time.time()-t,1))
"
```

The index caches to `story_mvp/faiss_index/<model_slug>.index` with a matching
`.signature`, reused as long as the document count and a SHA-256 content
signature both match. If it rebuilds on every start, the corpus file is changing
underneath it.

Report build wall time, vector count, and dimension.

## 7. Optional: re-ingest the PDFs

Only needed if you change or add source PDFs:

```bash
python -m story_mvp.ingest_curriculum --source story_datasets/NCERT_6th-8th \
  --output knowledge_base/processed --workers $(nproc)
```

This is `pypdf` text extraction — **CPU and IO bound, the GPU does not help.** Set
`--workers` to the core count. Metadata is inferred from folder and file names,
not PDF contents, so new PDFs must follow the existing naming convention or they
land in `unknown` buckets and metadata filters will not find them.

## 8. Run and verify

```bash
python -m story_mvp.app     # http://127.0.0.1:5050
```

To reach it from another machine, change the host in `run_story_studio()` to
`0.0.0.0` — trusted networks only. This is the Flask dev server with a hardcoded
`SECRET_KEY`; it is not hardened for public exposure.

```bash
# 1. Which engine is actually live?
curl -s http://127.0.0.1:5050/api/health
```

Expect `"engine": "free_first_hybrid_rag"` and a real `"model_provider"`. If you
see `"local_story_tutor_mvp"` or `"deterministic"`, the AI path failed at startup
and you are looking at template output.

```bash
# 2. A grounded generation
curl -s -X POST http://127.0.0.1:5050/api/generate \
  -H "Content-Type: application/json" \
  -d '{"idea":"Explain how energy transfers when a spoon warms in hot water",
       "class_level":"6","subject":"science","language":"english",
       "mode":"expand","output_type":"study_story"}'
```

```bash
# 3. Tests
python -m pytest tests/test_story_mvp.py tests/test_storytutor_options.py \
                 tests/test_curriculum_ingestion.py -q      # 10 tests, must pass
python -m pytest tests/test_rag_engine.py -q                # builds a small temp index
# test_llm_client.py and test_ai_integration.py auto-skip without GROQ_API_KEY
```

## 9. Acceptance criteria

Do not report success until all of these hold:

1. `torch.cuda.is_available()` is `True` and the embedding model reports a CUDA
   device at startup.
2. The full index builds in minutes, not hours, and reports 7,992 vectors.
3. `/api/health` returns `engine: free_first_hybrid_rag` with a
   non-deterministic `model_provider`.
4. A Class 6 / science / english generation returns a **non-empty
   `retrieved_sources`** array whose entries carry `source_file` and page numbers.
5. The same request returns non-empty `explanation`, `story`, `quiz`,
   `storyboard`, and `video_demo_plan`.
6. The response key set is **identical** to what the deterministic engine returns
   for the same payload.
7. The 10 offline tests pass.
8. A Marathi request returns Marathi sources, or English sources tagged
   `retrieval_language_fallback: "english"` — both are correct behaviour.

## 10. Known traps

- **Silent degradation is everywhere.** Four independent fallback layers mean the
  app never errors — it just quietly gets worse. Deterministic output is
  confident, well-formed prose about locked phones and floorboards (the project
  began as a fiction generator). Thriller imagery instead of a science
  explanation means the LLM path failed. Always check `model_provider`.
- **`RAG_TOP_K` is dead config** — nothing reads it. `top_k` is hardcoded to 3 at
  `rag_engine.py:106`.
- **`_safety_check()` is three substring comparisons** and self-documents as a
  demo filter. It is not moderation.
- **`output_type` and `difficulty` are transported but not honoured** — they reach
  the prompt, but no code branches on them. A `quiz` request will not return a
  quiz-shaped response.
- **`sys.path` is mutated in `app.py`.** Always run as a module
  (`python -m story_mvp.app`), never `python story_mvp/app.py`.

## 11. Report back

- GPU model, VRAM, driver, CUDA version, torch build.
- Index build wall time, vector count, dimension.
- The `.env` you ended up with, minus secrets.
- Which Ollama model you chose and whether it reliably produced parseable JSON.
- The three diffs from section 5, as a patch.
- `/api/health` output and one full `/api/generate` response.
- Test results.
- Anywhere the GPU did *not* help, so we know where the real bottleneck is.

## 12. Out of scope

Do not refactor the architecture, swap the web framework, restructure the
response contract, add a database, containerize, or clean up the deterministic
engine. Get it running fast and correctly on this hardware, verify it, report
back. Improvements are a separate conversation — `concept.md` §10 already has a
prioritized list.
