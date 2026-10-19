# research/ — turning StoryTutor-MM into measured research

Executable scaffold for [`RESEARCH_PLAN.md`](../RESEARCH_PLAN.md). Every item
maps to an artifact or command in this repository — plans without commands
drift, so each row here must stay runnable.

**Run everything on Sol** (`scg011:~/storytutor`) with the `storytutor`
environment active and the stack up (`bash scripts/sol.sh up`).

---

## Status board

| ID | Research question | Artifact in this repo | State | Command |
| --- | --- | --- | --- | --- |
| **R1** | Claim-level verification of generated explainer videos | `research/aggregate_video_claims.py` → claim/verdict dataset from video sidecars | **scaffolded — needs videos on Sol** | `python research/aggregate_video_claims.py` |
| **R2** | NCERT-Ground: trilingual groundedness & attribution benchmark | golden sets in `eval/golden/`; frozen baselines via `research/run_baselines.sh` | **scaffolded — needs baseline run** | `bash research/run_baselines.sh` |
| **R3** | Learning-gain RCT | school pilot protocol (external; needs consent + schools) | not started | — |
| **R4** | Code-switched / romanized query retrieval | `research/queries/romanized_queries.jsonl` (56-query seed set) + `research/run_r4_eval.py` | **runner written — needs Sol run** | `python research/run_r4_eval.py` |
| **R5** | Devanagari extraction quality | `story_mvp/deva_quality.py` + `eval/diagnose_corpus.py` | built, needs write-up | `python eval/diagnose_corpus.py` |
| **R6** | Model size vs teaching quality | roster in `model_clients.py`; judged by eval harness | not started | — |
| **R7** | Misconception-triggered retrieval | `knowledge_base/misconceptions/` | not started (needs A/B arm) | — |
| **R8** | Learner memory | — (product feature first) | not started | — |
| **R9** | Syllabus containment / refusal precision | refusal cases in `eval/chat_questions.json` | partial | `python eval/run_chat_tests.py --only refusal` |
| **R10** | Free-first systems numbers | latency fields already in every response | partial — needs a load test | `bash research/run_baselines.sh` |

---

## The three commands that produce this week's numbers

```bash
cd ~/storytutor && bash scripts/sol.sh up      # stack must be up first

bash research/run_baselines.sh                 # R2 + R10: frozen snapshot of
                                               # retrieval/RAG/chat metrics into
                                               # research/reports/<date>/

python research/run_r4_eval.py                 # R4: recall by query style
                                               # native vs romanized vs code-switched

python research/aggregate_video_claims.py      # R1: (claim, evidence, verdict)
                                               # dataset from every video made so far
```

### Baseline log

| Date | Result | Note |
| --- | --- | --- |
| 2026-10-19 | retrieval + chat valid; **rag_eval void** | eval shell lacked `OLLAMA_MODEL` → Ollama 404 → generation measured the passage fallback. Fixed: `run_baselines.sh` now sources `scripts/sol_env.sh` and writes `config.txt` into the snapshot; `run_rag_eval.py` loads `.env` like `app.py`. **Rerun before quoting any generation number.** |

## Data sources — where research numbers come from

| Data | Location | Produced by |
| --- | --- | --- |
| Video claim/verdict audits | `outputs/videos/*.json` (sidecars) | every successful `Make video` / `make_video.py` run |
| Golden set (268 verified) | `eval/golden/golden_set_verified.jsonl` | `eval/build_golden_set.py` + review |
| Behavioural chat results | `eval/reports/chat_test_report.json` | `eval/run_chat_tests.py` (53 questions) |
| Corpus quality | `eval/reports/corpus_damage.json` | `eval/diagnose_corpus.py` |
| Frozen metric snapshots | `research/reports/<YYYY-MM-DD>/` | `research/run_baselines.sh` |

## Ground rules (from `plan.md` §1 and the research risks)

1. **Freeze before comparing.** Any number quoted in a paper comes from a
   dated folder under `research/reports/`, never from a rerun.
2. **Judge ≠ generator.** Any teaching-quality scoring uses a different model
   than the one that produced the answer.
3. **No child data without consent.** R3/R8 logs are anonymized aggregates
   only (DPDP Act 2023 posture).
4. **Benchmark releases carry chunk IDs + offsets, not NCERT text.**

## Next artifacts to build (in order)

1. ~~`research/run_r4_eval.py`~~ **done** — run it on Sol for the first R4 numbers.
2. Sidecar claim export on **Sol** (run `aggregate_video_claims.py` after each
   demo day — the dataset grows with every video you make).
3. **Refusal false-positive set (R9):** `os06 "आज मुंबई में मौसम कैसा है?"`
   answered *as grounded* — vocabulary matches the weather chapter so the
   evidence gate lets it through. Turn all such cases into labelled negatives.
4. **Missing-source set (R2 negatives):** `hi05`, `mr08`, `xl01`, `xl03`
   found no sources — each needs a gold chunk id so recall and refusal can be
   scored on the same question.
5. R1 paper draft: judge/embedding disagreement analysis over the exported set.
