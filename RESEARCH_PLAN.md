# StoryTutor-MM → Research Agenda

**Thesis:** the system that answers NCERT Class 6–8 questions from the real
book — in English/Hindi/Marathi, with page-level citations and *fact-checked*
generated explainer videos — is itself a research instrument. This plan names
the novel, publishable questions it can answer, ranked by
(novelty × feasibility with artifacts already built).

**Inspiration:** `AI Layer & K12 Workflow.pdf` — approved-book-only retrieval →
page attribution → level-appropriate explanation → read-aloud → 30-second
narrated lesson → shared HPC, no school GPUs. Each step is an evaluable claim.

**Date:** 2026-10-18

---

## 0. Research assets already built (mostly unnamed)

| Asset | Where | Research value |
| --- | --- | --- |
| **Per-claim video fact-check**: decomposes a generated video's narration into claims `[C…]` *and* diagram edges `[D…]`, verifies each against retrieved evidence via LLM judge + embedding override (0.55/0.75 cos thresholds), degrades judge→embedding→keyword, writes verdict to sidecar JSON with a `verified` flag shown in UI | `video/script.py::grounding_problems` | **headline novel contribution** — nobody verifies generated educational video claim-by-claim |
| **268-question verified golden set** (keep_clean/keep_noisy/reject, reference answers, question types) + 53 behavioural chat questions | `eval/golden/`, `eval/chat_questions.json` | seed of a public trilingual benchmark |
| **Retrieval/RAG/chat eval harness** (recall@k, MRR, citation rate, invented citations, lexical groundedness, refusal rate, language match, latency) | `eval/run_*.py` | reproducible baselines, sliced by language |
| **Two-layer knowledge base**: 7,086 curriculum chunks + hand-written pedagogical seed incl. named **misconception cards** retrieved into the same context | `knowledge_base/` | misconception-aware retrieval experiments |
| **Bilingual chunk alignment** across en/hi/mar for the same NCERT chapter/page | `story_datasets/NCERT_6th-8th` | parallel curriculum corpus for MT/cross-lingual retrieval |
| **Devanagari damage metric + repair** (orphan matras/viramas, glyph intruders, 34,985→98; PyMuPDF-vs-pypdf selector) | `deva_quality.py`, `ingest_curriculum.py` | data-centric NLP artifact, no ground truth needed |
| **Query-rewrite rescue** (one LLM rewrite when evidence < 0.25) + Devanagari-aware BM25 + RRF hybrid | `rag_chat.py`, `bm25.py`, `reranker.py` | mechanism for code-switched retrieval study |
| **Full model roster at 4 sizes** (1.7B Ollama → 8B → 14B vLLM → 24B Sarvam opt-in) | `model_clients.py` | quality–cost Pareto study for Indic K-12 |

---

## 1. Tier 1 — publishable with data from the school pilot

### R1. Claim-level verification of generated multimodal explainer videos ⭐ lead paper
- **Question:** can a generated 30-second lesson video be *provably* limited to
  what its sources say — narration lines, diagrams, and cross-modal consistency?
- **Method:** formalize `grounding_problems` as claim extraction (text claims
  `[C…]` + **diagram-edge claims** `[D…]`) → evidence attribution → verdict;
  study judge/embedding disagreement, threshold sensitivity, and the
  **diagram entailment subtask** ("is edge *Hot water → Metal spoon* supported?").
- **Contribution:** first verification framework + released benchmark of
  **(claim, evidence, verdict) triples in English, Hindi and Marathi**, including
  cross-modal edges. Compare LLM-judge vs embedding vs lexical baselines — all
  three already implemented.
- **Venue:** AIED, EMNLP Findings, NeurIPS D&B track.

### R2. NCERT-Ground: a trilingual groundedness & attribution benchmark
- **Question:** how faithful is curriculum RAG in low-resource Indic languages,
  and can the system correctly *refuse* unanswerable questions?
- **Build from:** `golden_set_verified.jsonl` (268) + refusal/negative cases +
  page-level attribution labels (already in chunk metadata).
- **Metrics:** citation precision/recall, page-level attribution accuracy,
  groundedness, refusal correctness, **cross-lingual attribution** (Marathi
  question → English source page), sliced by language and class.
- **Baselines:** our pipeline + naive RAG (no gate/no rerank) + closed-book LLM.
- **Why novel:** existing faithfulness benchmarks (RAGAS, FaithDial, FreshQA)
  are English and general-domain; none target a K-12 textbook contract.
- **Venue:** NeurIPS/LREC-COLING datasets, EMNLP Findings.

### R3. Learning-gain RCT — the paper that also sells the product ⭐ highest impact
- **Gap:** `plan.md` §9 and `concept.md` §9 both admit *nothing measures whether
  a 12-year-old understood*.
- **Design:** pre-test → condition → post-test, **Hake-style normalized gain**,
  powered for 2 classes × ≥2 schools (the pilot itself), sliced by language and
  class level.
  - **A:** StoryTutor (grounded + video)
  - **B:** textbook-only (control)
  - **C:** generic closed-book LLM answer, no citations (ablation)
  - Optional **D:** grounded answer *without* misconception cards (bridges R7)
- **Analysis:** misconception-category error reduction (the product's core claim
  vs "a moving object needs a continuous force"), delayed retention if feasible,
  qualitative teacher interviews.
- **Ethics:** DPDP 2023 child-data posture + parental consent + anonymized
  aggregates; institutional affiliation needed for credibility.
- **Dual use:** the results **are** the school sales case study.

### R4. Code-switched & romanized query retrieval for Indic learners
- **Question:** real students type `"photosynthesis kya hai"`,
  `"spoon garam kyu hota hai"`, Hinglish in Latin script, or a Marathi question
  about an English-medium book. Which retrieval strategy survives?
- **Study:** native script vs romanized vs code-switched vs cross-lingual
  queries; ablate the one-shot **query-rewrite rescue** (when does rewriting
  help vs hurt?); ablate Devanagari-aware BM25 vs naive tokenization.
- **Metric:** recall@5 / chapter-hit by query style × target language.
- **Contribution:** first systematic robustness study for K-12 Indic search;
  a released query-perturbation set.

---

## 2. Tier 2 — deeper, longer horizon

### R5. Devanagari extraction quality without ground truth
`deva_quality.py` (orphan matra/virama detection, doubled-mark repair, glyph
intruder removal, PyMuPDF-vs-pypdf A/B selection; 34,985→98 damage) as a
*no-reference* text quality metric, validated by downstream retrieval
degradation. Resource/data-centric paper. Venue: LREC-COLING, ACL resource
track, Code4Soc. **Nearly free — just needs writing up.**

### R6. How small can a teaching model be? (Indic K-12 quality–cost Pareto)
1.7B / 8B / 14B / 24B across en/hi/mar judged on **teaching-quality** rubrics
(explanation depth, analogy presence, readability for class level, citation
validity) by a *different* judge model to limit self-preference. Output: the
frontier table that simultaneously justifies **pricing** and model tiers.
Ties to `plan.md` §6D (teaching evals, locally judged).

### R7. Misconception-triggered retrieval and prompting
Your two-layer KB is an untested hypothesis: does retrieving the *named
misconception card* alongside the textbook page reduce post-test errors vs
plain RAG? A/B within R3 (arm D) or a lab study. Uses
`knowledge_base/misconceptions/common_misconceptions.json` as-is.

### R8. Learner memory / adaptive retrieval
Store quiz outcomes per learner; feed miss history into retrieval filters and
prompt context (`concept.md` §10.5). Measure cold-start vs adapted performance,
plus a privacy-preserving variant (on-device history, DPDP-compliant). Turns the
generator into a tutor — engineering roadmap item and evaluable research.

### R9. Syllabus containment & child-safe refusal for minors
Detect out-of-syllabus and harmful requests from 11–14-year-olds; respond
"not in your book — the nearest chapter is X." Evaluate refusal precision
across three languages; extend `guardrails.py` substring checks toward
measured safety. DPDP-compliance artifact. Venue: AIES / safety workshops.

### R10. Free-first systems paper (AI for under-resourced schools)
"Zero cost of goods": open weights only, no API keys, one GPU, queued media,
shared across a whole class — the PDF's step 6 as a systems artifact/demo
paper, with measured latency, cost-per-answer, and classroom-scale load
results. Artifact-evaluation track (AIED/EDM demos, MLSys poster).

---

## 3. Recommended portfolio & sequencing

1. **Now (no IRB, artifacts exist):** write **R1** (video claim verification)
   and stand up **R2** (benchmark release). These are the novelty.
2. **During the school pilot (consented):** collect **R3** learning-gain data —
   the pilot is designed to produce the paper. Add **R4** query-logging study
   from the same logs.
3. **Cheap side-projects:** **R5** (write up what's built), **R6** (run the
   roster you already serve).
4. **After learner memory ships:** **R7/R8/R9**.

Venues to target by deadline: AIED (early Dec), LAK/EDM (Jan), EMNLP Findings,
NeurIPS D&B (May), LREC-COLING. arXiv preprints immediately for both sales and
citations.

## 4. Definition of done (research)

1. `NCERT-Ground` publicly released (questions + page-level labels + metrics
   code), with our system as a documented baseline in all three languages.
2. R1 method integrated so every produced video carries a machine-checkable
   verification report (already in the sidecar — now with published numbers).
3. R3 completes with a powered effect size, published + usable as the school
   sales case study.
4. Every claim quoted in marketing ("answers only from the book, shows the
   page") is backed by a published measurement.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| No institutional affiliation for RCT | partner with an education dept./college; else frame as classroom-based practitioner study |
| Small pilot underpowers R3 | two schools × two classes; preregister; report CIs, not p-values |
| Benchmark release needs NCERT redistribution clearance | release questions + chunk IDs + offsets; instruct users to ingest their own NCERT PDFs (code already does) |
| Judge-model bias in teaching-quality scoring | different judge than generator (`plan.md` §6D); validate against teacher ratings on a 50-item sample |
| Concept drift as product evolves | freeze eval harness versions per paper |
