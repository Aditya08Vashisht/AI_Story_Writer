# StoryTutor-MM → Product: Selling to Schools

**Goal:** turn the ASU-Sol research prototype into a multi-tenant, cloud-hosted
tutor that Indian schools (Class 6–8, NCERT, English/Hindi/Marathi) pay for —
sold on the four promises from `AI Layer & K12 Workflow.pdf`:

1. Answers **only from the approved textbook**, with the **exact page** shown.
2. Explained **at the student's level**, in their language.
3. **Read aloud**, plus a **fact-checked ~30-second lesson video**.
4. Schools **buy no GPUs** — heavy lifting runs on shared cloud compute.

**Date:** 2026-10-18 · supersedes all earlier product thinking.

---

## 0. What exists today (honest inventory)

The repo is **not** what `concept.md` describes — several sections are stale.
Verified against code:

| Capability | Status | Evidence |
| --- | --- | --- |
| Grounded chat with `[S1]` page citations, SSE streaming | **working** | `story_mvp/rag_chat.py`, `static/chat.html` |
| Hybrid retrieval: dense + BM25 → RRF k=60 → optional cross-encoder rerank | **working** | `rag_engine.py`, `bm25.py`, `reranker.py` |
| Evidence gate + one query-rewrite rescue + honest refusal | **working** | `rag_chat.py::retrieve_with_rescue`, MIN_EVIDENCE_SCORE 0.25 |
| Guardrails: injection, Indian-PII redaction, citation validation, lexical groundedness | **working** | `guardrails.py` |
| Trilingual TTS (Parler `indic-parler-tts` → MMS fallback) on :5060 | **working** | `scripts/tts_server.py`, `story_mvp/video/tts.py` |
| Real video rendering: FLUX.1-schnell images + narration + burned Devanagari subtitles + **per-claim fact-check** | **working** | `story_mvp/video/` (pipeline, script, images, cards, render) |
| Eval harness: 268-question golden set, retrieval/RAG/chat evals | **built** | `eval/` (reports currently live only on Sol) |
| LLM on ASU Sol: vLLM `Qwen/Qwen3-14B` bf16/8k on 1× A100; Ollama `qwen2.5:7b` fallback; Groq free tier first | **working on borrowed compute** | `slurm/serve_vllm.sbatch`, `scripts/sol.sh` |
| Auth / multi-tenancy / per-school data | **absent** | — |
| Learner memory, teacher dashboard, progress reports | **absent** | `concept.md` §9 |
| `output_type` / `difficulty` honoured | **no** — transported only | `generator.py` |
| Legacy story generator behind `/api/generate` | **still present** (UI can't reach it) | `generator.py`, tests lock contract |

**Fix-before-quoting numbers:** corpus is **7,086 chunks** (counted directly).
`concept.md` says 7,992; `HOW_IT_WORKS.md` says 7,095. One number everywhere.

**Known bugs to fix immediately:**
- `/api/options` (`app.py` L299–313) references unimported `MODE_TITLES/GENRES/TONES` → guaranteed HTTP 500.
- `README.md` still contains unresolved `<<<<<<< HEAD` merge-conflict markers.
- `app.config["SECRET_KEY"]` hardcoded (`"story-tutor-mm-mvp-2026"`).

---

## 1. The legal blocker — do this first, before writing product code

Selling this exposes three licensing issues that a demo does not:

| Asset | License | Problem | Action |
| --- | --- | --- | --- |
| `facebook/mms-tts-{hin,mar,eng}` | **CC-BY-NC-4.0** | Non-commercial — **cannot ship in a paid product**. The code even says "fine for research". | Replace with a commercially-clean engine (e.g. AI4Bharat Indic TTS under its actual terms, Kokoro, or a licensed Indian TTS) — or keep MMS only as a self-hosted, disclosed research fallback. |
| `ai4bharat/indic-parler-tts` | documented Apache-2.0 **in our code** | Must verify the **model weights** license, not just repo code. Parler-family weights have historically been CC-BY-NC — confirm for the AI4Bharat port. | Written confirmation; if NC → replace or negotiate. |
| **NCERT PDFs** | NCERT copyright | Extracting and serving textbook text to **paying** customers likely requires permission. | (a) Seek NCERT / district authorization. (b) Build ingestion to accept **district-provided corpora** (the PDF already says "or ones a district provides") — this both diversifies content and de-risks NCERT. (c) Until cleared, pilot with partner schools under an educational-use MOU. |
| Qwen3, FLUX.1-schnell, bge-m3, bge-reranker-v2-m3 | Apache-2.0 / MIT | clean | none |
| Devanagari fonts (Noto) in burned subtitles | OFL | clean | none |

**The DPDP Act 2023 (India) angle — turn compliance into the pitch:**
- Children's data ⇒ verifiable parental consent, no behavioural tracking/ profiling, no targeted advertising.
- We already redact phone/email/Aadhaar patterns (`guardrails.py`), and questions never leave the deployment region.
- **Sales line:** *"Your child's questions never leave the country, never train anyone's model, and are never tracked."* No US/UK competitor can say this.

---

## 2. Deployment architecture — what to buy

### 2.1 Two workloads, two GPU strategies

| Workload | Latency need | GPU requirement |
| --- | --- | --- |
| **Interactive chat:** retrieval (bge-m3 ~2 GB) + reranker (2.2 GB, opt-in) + LLM generation | first token < 1 s | **24 GB is sufficient** with vLLM `Qwen3-8B`; `Qwen3-14B` bf16 needs ~40 GB |
| **Media:** FLUX.1-schnell (~12–24 GB) + Parler TTS (~4–8 GB) + ffmpeg encode | 2–5 min, async, already a job queue | **separate, batchable, preemptible/spot GPU** |

Do **not** put both on one card: a video job would starve the chat path.

### 2.2 Recommended shape: your hosted multi-tenant SaaS

One cluster serves many schools; schools own nothing. (On-prem per school is
support-death for a solo founder and Indian schools have weak IT capacity —
revisit as an "enterprise/offline" tier later.)

```
                    ┌────────────────────────────────────────────┐
 schools ──HTTPS──► │  App tier (2× containers)                  │
 (browser, PWA)     │  Flask + auth + tenancy + /api/chat/stream │
                    └──────────────┬─────────────────────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                               ▼
 ┌──────────────┐        ┌──────────────────┐          ┌────────────────────┐
 │ Object store │        │ Interactive GPU  │          │ Media workers      │
 │ per-tenant   │        │ L4/A10G 24 GB    │          │ spot L4/A10G       │
 │ videos,      │        │ vLLM Qwen3-8B    │          │ FLUX + TTS + ffmpeg │
 │ index, cache │        │ bge-m3 (+rerank) │          │ drains /api/video  │
 └──────────────┘        └──────────────────┘          │ queue              │
                                                       └────────────────────┘
```

**Rough per-school economics** (20 schools × ~300 students sharing one
interactive node + queued media on spot):
- Interactive node dedicated ≈ **$0.6–0.9/hr** (L4/A10G on RunPod/Lambda/FluidStack) ≈ ₹15–18k/month
- Media on spot, batched ≈ **₹5–10k/month** at pilot volume
- ⇒ infra ≈ **₹15–25/student/month** — far below a tuition class, your pricing anchor.

**The code already supports this** — no re-architecture needed:
- `VLLM_BASE_URL` / `VLLM_MODEL` point the provider chain at any OpenAI-compatible server (`model_clients.py::VLLMClient`).
- `/api/video` is already a queue + worker (`app.py::_video_worker`) — just externalize the queue (Redis/RQ or a DB table) for horizontal workers.
- `STORYTUTOR_TTS_URL` makes TTS a service, already remote-capable.

### 2.3 Hardening roadmap (engineering work list)

**P0 — cannot sell without these (≈2–3 weeks)**
1. **License swaps** (MMS TTS, Parler weights confirmation, NCERT permission path).
2. **Auth + tenancy:** school accounts, **class codes** for students (no emails — they're 11), per-tenant video library + object storage, per-tenant rate limits; real `SECRET_KEY`.
3. Fix `/api/options` 500, README conflicts; quarantine legacy story path (`/api/generate`) or brand it clearly as internal API.
4. **Pre-build the index as a versioned artifact** (kill silent import-time cold start; concept.md §8 already flags this) + a readiness endpoint `/api/health` must gate on it.
5. Containerize: `Dockerfile` + compose for app / vllm / tts / media-worker; one-command deploy replacing `sol.sh` gymnastics.

**P1 — what schools actually buy (≈1 month)**
6. **Teacher surface:** assign a chapter; see top misconceptions the class asked about; print the quiz; see per-student library. (Teachers adopt; students use.)
7. **Quality-tier dashboard:** surface `model_provider` / `retrieval_method` / groundedness / refusal rate per tenant — concept.md §3.3's admitted cost of silent degradation, now with a support contract attached. Silent degradation = angry school, not quiet degradation.
8. **Learner memory:** store quiz misses per learner (consented), feed into retrieval filters + prompt context (`concept.md` §10.5). Turns a generator into a tutor and powers progress reports.
9. Commit eval reports into the repo; run `run_chat_tests`'s 53 questions as a nightly smoke test per environment.

**P2 — scale & polish**
10. IVF/HNSW or prebuilt FAISS artifact (currently `IndexFlatIP`, O(n), fine to ~10⁵ chunks; you have 7k — headroom exists, do it when >50k).
11. Observability (per-request tier/latency/groundedness), backups, per-tenant usage metering.
12. Honour `output_type`/`difficulty` (`generator.py` has no branch today).

### 2.4 Go-to-market

- **Pilot free, 1 term, 2 schools** (one Marathi-medium, one English-medium) → gather learning-gain evidence (see research plan R3) → convert to paid on *your own data*.
- Sell to **schools**, deliver value to **teachers** first.
- Lead differentiator (four things generic AI-tutor competitors cannot say):
  > *Answers only from **your** book · shows the **page** · works in **Marathi** · nothing leaves **India**.*
- Later tiers: district deployment (district-supplied corpus), offline-cache hybrid for low-connectivity schools, assessment/reporting add-on.

---

## 3. Definition of done (product)

1. A Class 7 student on a school Chromebook/phone, anywhere in India, gets a Marathi answer with a clickable page citation, first token < 1 s, **on your cloud, not ASU**.
2. Zero NCERT/commercial-license exposure; DPDP posture documented; parent-facing privacy one-pager exists.
3. A teacher can log in, see what the class asked, and print a quiz — without IT help.
4. Schools never see a silent quality drop: the health card tells them which model tier is serving.
5. Unit economics hold at ≥20 schools on one interactive node; media queue drains on spot.

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| NCERT won't license / district corpora hard to get | district-provided ingestion path; frame as "bring your textbook" |
| Parler/MMS both NC → no clean Hindi/Marathi voice | evaluate Indic-TTS/Kokoro/licensed options now; voice is a core PDF promise |
| One founder maintaining infra + sales | maximize managed services (vLLM containers, managed Redis/K8s); P0 minimalism |
| Schools don't pay without proof | the RCT (research R3) is the proof; design pilot to produce it |
| Internet outages kill SaaS | cached corpus + queued sync; later offline tier |
