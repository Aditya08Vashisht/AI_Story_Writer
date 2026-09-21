# Skills — StoryTutor-MM

What you need to know to work on this codebase, where each skill actually shows
up, and in what order to pick things up. Written for someone joining the project
or auditing what it demands.

---

## Skill map at a glance

| Domain | Weight | Depth needed |
| --- | --- | --- |
| Python backend engineering | ●●●●● | Intermediate |
| RAG / vector retrieval | ●●●●● | Intermediate–advanced |
| Prompt engineering & LLM integration | ●●●●○ | Intermediate |
| Data engineering (PDF → structured) | ●●●○○ | Intermediate |
| Frontend (vanilla JS/CSS) | ●●●○○ | Beginner–intermediate |
| Testing & reliability engineering | ●●●○○ | Intermediate |
| Multilingual / Indic NLP | ●●○○○ | Awareness |
| Learning design & pedagogy | ●●●○○ | Domain literacy |
| DevOps / packaging | ●●○○○ | Beginner |

---

## 1. Python backend engineering

**Level: intermediate. Non-negotiable — every module is Python.**

Concretely used in this repo:

- **Frozen dataclasses** — `StoryRequest` in
  [generator.py:96](story_mvp/generator.py#L96) and `PdfMetadata` in
  [ingest_curriculum.py:30](story_mvp/ingest_curriculum.py#L30). Immutable value
  objects carried through the pipeline.
- **Inheritance for behavioural specialization** — `RetrievalService(StoryRAG)`
  overrides only `retrieve()` and delegates upward with
  `super().retrieve(*args, **kwargs)`
  ([rag_engine.py:231](story_mvp/rag_engine.py#L231)).
- **Adapter / strategy pattern** — `BaseModelClient` with four concrete
  implementations plus a `FallbackModelClient` composite that iterates them
  ([model_clients.py](story_mvp/model_clients.py)).
- **Layered exception handling** — try/except at import time, per request, and
  per provider. Reading this code means tracing *which* handler catches what.
- **Flask** — routing, `render_template`, `jsonify`, Jinja2 loops in
  [templates/index.html](story_mvp/templates/index.html).
- **`from __future__ import annotations`** and `typing` — used consistently.
- **`concurrent.futures.ProcessPoolExecutor`** — parallel PDF extraction, with
  the correct `finally: shutdown(wait=True, cancel_futures=True)`.
- **Atomic file writes** — `.tmp` + `os.replace()` in `_write_json()`, the reason
  a killed ingestion never leaves a corrupt corpus.
- **`os.walk`, `os.path`, cross-platform path handling** — the ingester
  normalizes `\` to `/` before parsing, because paths here are Windows.

**How to build it:** read [generator.py](story_mvp/generator.py) end to end. It
exercises nearly all of the above in 490 lines.

---

## 2. RAG and vector retrieval

**Level: intermediate to advanced. The technical core of the project.**

### 2.1 Embeddings

You need to understand what a sentence embedding *is* and why model choice
dominates behaviour here. The default `BAAI/bge-m3` is chosen for multilingual
capability — it must place a Marathi science passage near an English query about
the same concept. `all-MiniLM-L6-v2` (used in tests) is English-centric and much
smaller. Swapping the model changes retrieval quality, index size, build time,
and the on-disk cache filename.

### 2.2 FAISS

- `IndexFlatIP` — exhaustive inner-product search.
- `faiss.normalize_L2()` before adding and before querying — this is what makes
  inner product equal cosine similarity. Miss it and scores are meaningless.
- `index.search(q, k)` returning `(distances, indices)`, with `-1` marking
  padding when fewer than `k` results exist.
- `write_index` / `read_index` persistence, and cache invalidation via a
  content signature ([rag_engine.py:91](story_mvp/rag_engine.py#L91)).

### 2.3 Hybrid search

`_hybrid_score()` blends dense similarity with lexical overlap at 85/15. You
should be able to argue about that ratio: why dense alone under-retrieves exact
technical terms, and why lexical alone fails across languages.

### 2.4 Metadata filtering — the hard part

Three ideas here are genuinely non-obvious and worth internalizing:

1. **Flat indexes have no metadata predicates.** When filters are active the code
   sets `fetch_k = len(documents)` and filters afterwards, because filtering
   *after* a top-k cut silently drops correct results
   ([rag_engine.py:120-124](story_mvp/rag_engine.py#L120-L124)).
2. **Filter negotiation.** `_supported_filters()` only applies a filter the
   corpus can actually satisfy — a learner's natural-language chapter name must
   not zero out the result set when the corpus uses numeric chapter ids.
3. **Language fallback.** `RetrievalService.retrieve()` back-fills thin
   Hindi/Marathi results with English and tags them, trading perfect
   language-match for non-empty grounding.

**How to build it:** work through [test_rag_engine.py](tests/test_rag_engine.py).
Each test isolates one of these behaviours.

---

## 3. Prompt engineering and LLM integration

**Level: intermediate.**

- **Structured output.** The system prompt embeds a full JSON schema inline
  ([model_clients.py:177](story_mvp/model_clients.py#L177)). Understand why
  provider-level enforcement (`response_format={"type":"json_object"}` for Groq,
  `"format": "json"` for Ollama) is used *in addition to* schema-in-prompt, and
  why `parse_json_response()` still needs a fenced-block retry.
- **Grounding instructions.** "Do not invent textbook facts if the context is
  insufficient; say what needs verification" — anti-hallucination phrasing that
  gives the model a licence to admit uncertainty.
- **System vs user split.** Role, constraints, grounding, and schema in system;
  the concrete request in user (`build_user_prompt()`).
- **Multi-provider portability.** The same two messages must work across Groq
  (SDK), Ollama (`urllib` to `/api/chat`), and Sarvam (`urllib` to an
  OpenAI-shaped endpoint). Knowing where those three response envelopes differ is
  a real skill: `completion.choices[0].message.content` vs
  `body["message"]["content"]` vs `body["choices"][0]["message"]["content"]`.
- **Defensive merging.** `_educational_artifacts()` fills every missing field
  with a deterministic default, so partial model compliance still yields a
  complete response.
- **Parameter intuition** — `temperature=0.6`, `max_tokens=1800`: high enough for
  narrative variety, bounded enough to keep JSON well-formed.

---

## 4. Data engineering: PDF → structured corpus

**Level: intermediate.**

- **`pypdf` text extraction** and its limits — no OCR, so a scanned or
  image-heavy page yields empty text and is skipped silently.
- **Chunking strategy.** Fixed 1,800-character windows with 250-character
  overlap ([`_split_text`](story_mvp/ingest_curriculum.py#L203)). You should be
  able to explain the overlap (a concept split across a boundary stays retrievable
  from at least one chunk) and the weakness (splits mid-sentence, ignores
  headings and tables).
- **Metadata inference from filesystem structure.** Regex over the path to pull
  class, subject, and language; regex over the filename to pull chapter
  (`fmrcu101.pdf` → `101`). Brittle by design, cheap by design.
- **Pipeline hygiene.** Checkpointing every N files, atomic writes, per-file
  error capture into an audit rather than an aborted run.
- **Coverage auditing.** `_missing_expected_coverage()` enumerates all 18
  class × subject × language buckets and reports which are absent — a data
  completeness test, not a code test.
- **Encoding discipline.** `encoding="utf-8"` and `ensure_ascii=False`
  everywhere, because the corpus is majority Devanagari.

---

## 5. Frontend

**Level: beginner–intermediate. Vanilla only — no framework, no build step.**

- ES6+ async/await, `fetch`, DOM manipulation, `document.getElementById`
  binding into an `els` registry.
- **XSS awareness.** `escapeHtml()` routes every interpolated value through a
  detached `div.textContent` before it reaches `innerHTML`
  ([story.js:335](story_mvp/static/story.js#L335)). Any new rendering code must
  keep doing this — model output is untrusted input.
- **Offline-capable design.** The whole `generateLocalStory()` path mirrors the
  Python deterministic engine in JavaScript so `demo.html` works from a file://
  URL. Changing one engine without the other creates divergence.
- **Jinja2 templating** — [index.html](story_mvp/templates/index.html) loops over
  server-provided genres, tones, and modes.
- **CSS** — a single hand-written [story.css](story_mvp/static/story.css) using
  custom properties and grid.
- **Accessibility basics** — `role="tablist"`, `aria-live="polite"`,
  `aria-label` are present and should stay present.
- **Node.js (light)** — [serve_static.js](story_mvp/serve_static.js) is a ~50-line
  static server with a path-traversal guard (`filePath.startsWith(root)`).

---

## 6. Testing and reliability engineering

**Level: intermediate.**

- **pytest fixtures with scope** — `@pytest.fixture(scope="module")` so an
  expensive FAISS index is built once per module.
- **`tmp_path_factory`** — [test_rag_engine.py](tests/test_rag_engine.py) builds a
  7-document representative sample (one per class/subject/language collection)
  into a temp dir instead of indexing all 7,992 chunks.
- **Conditional skipping** — `pytest.skip()` when `GROQ_API_KEY` is absent, so
  the suite stays runnable offline.
- **Test isolation** — separate index directories, cleaned in fixture teardown,
  so tests never touch the production cache.
- **Contract testing** — asserting *response shape parity* between two
  independent engines is the most valuable pattern in this repo.
- **Behavioural assertions for non-deterministic systems** —
  `test_ai_output_is_not_deterministic_template` asserts the AI output does *not*
  contain known template phrases. You cannot assert exact LLM output; you can
  assert what it must not be.

---

## 7. Multilingual and Indic NLP awareness

**Level: awareness, not deep expertise.**

- Devanagari handling end to end: UTF-8 encoding, `ensure_ascii=False` JSON,
  fonts in the browser.
- Why a multilingual embedding model is required, not optional — the corpus is
  ~2/3 Hindi and Marathi by chunk count.
- Code-mixing: `hinglish` is a first-class language option in the UI but is
  deliberately *not* a retrieval filter value
  ([generator.py:153](story_mvp/generator.py#L153)) — there is no Hinglish
  corpus, so it maps to English for retrieval while still shaping generation.
- NCERT structure literacy: chapter numbering conventions, and the file-naming
  patterns (`fmrcu101`, `fhes114`, `gmsc101`) the ingester's regex depends on.

---

## 8. Learning design and pedagogy

**Level: domain literacy. Easy to under-invest in, and the thing that decides
whether the product is good.**

The knowledge base encodes real instructional-design concepts, and extending it
means understanding them:

- **Learning objectives** — observable, specific, assessable
  (`knowledge_base/curriculum/learning_objectives.json`).
- **Misconception-driven teaching** — each entry pairs a misconception with a
  correction and a diagnostic move
  (`knowledge_base/misconceptions/common_misconceptions.json`). This is the
  highest-leverage pedagogical asset in the repo.
- **Narrative pedagogy** — why a story makes an abstract concept retrievable, and
  how to keep the narrative from burying the concept.
- **Formative assessment** — quiz items with distractors that *diagnose* rather
  than merely score; note that every quiz item carries an `explanation` field.
- **Storyboarding for instruction** — the four-scene arc
  (goal → story → diagram → check) in `_storyboard()` is a lesson structure, not
  a UI decision.
- **Age-appropriateness** — Class 6–8, roughly ages 11–14, drives vocabulary,
  sentence length, and analogy choice.

---

## 9. DevOps and packaging

**Level: beginner, but with real gaps to close.**

Present: virtualenv workflow, `requirements.txt`, `python-dotenv` config,
`.gitignore` protecting `.env` and `faiss_index/`, a health endpoint.

Not present, and needed for anything beyond a demo: a production WSGI server,
containerization, a pre-built index artifact instead of build-on-boot,
`SECRET_KEY` from the environment, structured logging (the code uses `print`),
CI, and dependency pinning to exact versions.

---

## 10. Suggested onboarding order

1. **Run the static demo.** `story_mvp/demo.html` in a browser. Understand the
   product in two minutes with zero setup.
2. **Read [generator.py](story_mvp/generator.py) end to end.** It is the spine:
   request contract, both engines, response contract.
3. **Run the offline tests.**
   `pytest tests/test_story_mvp.py tests/test_storytutor_options.py tests/test_curriculum_ingestion.py -q`
   — 10 tests, under a second, no downloads.
4. **Read [rag_engine.py](story_mvp/rag_engine.py) with
   [test_rag_engine.py](tests/test_rag_engine.py) side by side.** The tests
   explain the non-obvious retrieval decisions.
5. **Run the ingester on a handful of PDFs:**
   `python -m story_mvp.ingest_curriculum --limit 3 --output <scratch dir>`
   then read the chunks and the audit.
6. **Start the Flask app** with `STORYTUTOR_EMBEDDING_MODEL=all-MiniLM-L6-v2`,
   and watch `/api/health` to see which engine is live.
7. **Read [model_clients.py](story_mvp/model_clients.py)** last — the prompt is
   where product intent and model behaviour meet, and it makes most sense once
   you know what fills the response.

---

## 11. Skill gaps visible in the current code

Honest assessment of where the project's own skill coverage is thin. Each is an
opportunity for whoever picks it up:

| Gap | Evidence | Skill to apply |
| --- | --- | --- |
| No RAG evaluation | Tests assert shape, never groundedness | Retrieval eval — recall@k, citation-support checking, a golden question set |
| No observability | `print()` statements, no request logs, no latency metrics | Structured logging, tracing |
| Blocking startup | Index built synchronously at import | Async init, readiness probes, pre-built artifacts |
| Frontend under-renders the API | `renderTutorExtras()` collapses storyboard/quiz/diagram into `<li>` lines | UI/UX and information design |
| Placeholder safety | Three substring checks | Content moderation, age-appropriateness policy |
| Unhonoured parameters | `output_type` and `difficulty` reach the prompt but branch nothing | Product logic design |
| No learner state | Every request is stateless | Data modelling, session/learner memory |
| O(n) retrieval | `IndexFlatIP` over 7,992 vectors, growing | ANN indexing — IVF, HNSW, quantization |
| Duplicated engines | Deterministic prose exists in both Python and JS and can drift | Shared-contract design, codegen or a single source of truth |
