# StoryTutor-MM, explained simply

*A plain-language guide to what this project is and how it works.*
*If `plan.md` is the engineering blueprint, this is the tour.*

---

## 1. The problem we're solving

Imagine you're in Class 7, sitting with your Science textbook, and you hit this:

> *"Energy is transferred from one object to another."*

You read it. You can even repeat it in an exam. But do you actually **get** it?
Probably not — because the sentence tells you a fact without showing you anything.

Now imagine you could ask a tutor:

> *"Explain how energy transfers when a spoon warms up in hot water."*

…and get back a clear explanation, a short story that makes it stick, a quick
quiz to check you understood, and a 30-second video — **in Hindi, Marathi, or
English**, whichever you think in.

That's what we're building. A tutor that never gets tired, never runs out of
patience, and costs nothing to run.

---

## 2. Why not just use ChatGPT?

Fair question. Three reasons.

**It makes things up.** Ask a general AI about a specific NCERT chapter and it
will answer confidently — sometimes with facts that aren't in your textbook at
all. In AI research this is called **hallucination**. For a student studying for
an exam, a confident wrong answer is worse than no answer.

**It doesn't know *your* book.** There are many science textbooks in the world.
You're being examined on one specific NCERT chapter. A general AI doesn't know
which one you're on.

**It costs money, and it's weakest exactly where we need it most.** Most AI
services charge per question, and most are trained overwhelmingly on English.
Their Hindi is okay. Their Marathi is usually worse.

So instead of asking an AI to *remember* your textbook, we built a system that
makes it **read your textbook before answering**.

---

## 3. The big idea: RAG

**RAG** stands for **Retrieval-Augmented Generation**. Three words, one idea:

> **Retrieval** — go find the right pages
> **Augmented** — hand those pages to the AI
> **Generation** — let it write an answer *using only those pages*

Think of the difference between two students in an exam:

| Student A | Student B |
|---|---|
| Answers from memory | Opens the textbook, finds the right page, then answers |
| Sounds confident | Sounds confident **and** can point to where it says so |
| Sometimes invents things | Can only use what's actually on the page |

Plain AI is Student A. **RAG is Student B.** That's the entire concept.

---

## 4. How our system actually works

Six steps, from your question to your answer.

### Step 1 — Build the library

We took **282 NCERT PDF textbooks** — Class 6, 7 and 8, Science and Social
Science, in English, Hindi and Marathi — and chopped them into **7,992 small
pieces** called *chunks*. Each chunk is roughly a page-worth of text.

Why chop them up? Because when you ask about photosynthesis, you want the two
paragraphs about photosynthesis — not an entire 40-page book.

Each chunk remembers where it came from: which class, subject, language,
chapter and page. That's how the system can later say *"this came from Class 6
Science, Chapter 8, page 3."*

### Step 2 — Give every chunk a "meaning fingerprint"

Here's the clever part.

A computer can't read meaning the way you do. So we use a model called
**bge-m3** to convert every chunk into a long list of numbers — about 1,024 of
them. This list is called an **embedding**, and you can think of it as the
chunk's *coordinates on a giant map of ideas*.

On this map, chunks with similar meanings sit close together. A paragraph about
photosynthesis in English lands near a paragraph about प्रकाश संश्लेषण in
Hindi — **even though they share no common words** — because they mean the
same thing.

That's the magic: the map is built on *meaning*, not spelling.

When you ask a question, we put your question on the same map and look at what's
nearby. This is called **semantic search**.

### Step 3 — Also search the old-fashioned way

Meaning-search is powerful but slightly blurry. If you search for
"photosynthesis," it might return chunks about plants in general, since they're
all nearby on the map.

So we run a **second, completely different search** at the same time, using a
method called **BM25**. BM25 doesn't care about meaning at all — it just counts
exact word matches, like `Ctrl+F` but smarter about which words matter. (Rare
words like "photosynthesis" count for a lot. Common words like "the" count for
almost nothing.)

Two searches, two different strengths:

| Search | Good at | Bad at |
|---|---|---|
| **Semantic** (bge-m3) | Understanding what you *meant* | Exact technical terms |
| **BM25** | Exact words, precise terms | Understanding meaning |

Together, they cover each other's blind spots.

> **A bug worth telling you about.** Our first version of BM25 broke Hindi and
> Marathi completely. The code used a rule that splits text at every non-letter
> character — but in Devanagari, the vowel marks (मात्रा) technically count as
> "not letters." So **प्रकाश** was being chopped into **7 separate pieces**
> instead of staying one word. Sparse search would have been silently useless on
> two-thirds of our books. A test caught it before it shipped. This is why you
> write tests.

### Step 4 — Combine the two lists fairly

Now we have two ranked lists of chunks. How do we merge them?

The naive approach is to average the scores. But that fails badly, because the
two searches produce scores on totally different scales — like trying to average
a cricket score with a temperature. The numbers just aren't comparable.

So we use **Reciprocal Rank Fusion (RRF)**, which throws the scores away
entirely and looks only at **position**:

> 1st place gets the most points. 2nd place gets fewer. And so on.
> A chunk that finishes high on **both** lists beats a chunk that tops only one.

It's exactly how you'd combine two judges' rankings in a competition — you don't
need the judges to agree on a scoring scale, only on an order.

### Step 5 — Have an expert double-check the top few

Both searches so far work the same way: they turn the question into numbers,
turn each chunk into numbers, and compare. Fast — but they never actually look
at the question and the chunk *together*.

So for the top 30 results, we bring in a **reranker** (`bge-reranker-v2-m3`).
This model reads your question and a chunk **side by side, at the same time**,
and judges: *does this passage actually answer this question?*

It's far more accurate — and far slower. Which is exactly why it only sees the
30 finalists, never all 7,992. It's the difference between a first-round filter
and a final interview.

This is usually the single biggest quality improvement in a RAG system.

### Step 6 — Write the answer, using only those pages

The top 5 chunks go to a language model (**Qwen3**, running on our own GPU) with
strict instructions:

- Explain in the student's language, at Class 6–8 level
- **Every factual sentence must cite its source**, like `[S1]` or `[S2]`
- **If the pages don't cover something, say so** — don't fill the gap from memory

That last rule is the anti-hallucination rule, and it's the whole point. The AI
isn't allowed to be Student A.

---

## 5. Why three languages is genuinely hard

This is the most interesting research problem in the project.

### Problem 1: The textbooks are damaged before we even start

The Marathi PDFs don't extract cleanly. Devanagari letters combine into
conjuncts — क् + ष becomes क्ष — and the PDF reader often breaks them apart,
inserting spaces in the middle of words. Real example from our data:

```
Should be:  तुम्हाला वाटते का
We got:     तुम् हे ा ली ा वी ाट ते क ा
```

We measured this properly. Out of 50 sample questions pulled from the books:

| Language | Usable | Main problem |
|---|---|---|
| English | **89%** | minor |
| Marathi | **45%** | letters physically broken apart |
| Hindi | **36%** | text fine, but context cut off mid-question |

Notice Hindi and Marathi fail for **different reasons**. That matters, because
they need different fixes — and it would be easy to lump them together and fix
the wrong thing.

### Problem 2: You can't fix bad data with a better model

This is the most important lesson in the whole project.

It's tempting to say *"Marathi is weak — let's use a Marathi-specific AI model!"*
But if the text in our library is already broken, a better model just reads the
broken text more carefully. **Garbage in, garbage out.**

So the plan fixes things in strict order:

1. **Fix the data** (re-extract the PDFs properly)
2. **Then** improve the searching
3. **Then** consider better models

Skipping to step 3 would let you *think* you improved things when you didn't.

### Problem 3: Uneven coverage

Some chapters have richer English material than Marathi. So when a Marathi
student asks a question, we plan to **also** translate it to English and search
both — using a small translation model (**IndicTrans2**) — then merge the
results. Better to find a good English source than nothing at all, as long as
we're honest about where it came from.

---

## 6. How do we know it actually works?

Here's the uncomfortable truth we're not hiding: **we don't know yet.**

We've tested English. We have **not** tested a single Hindi or Marathi question.
So any claim about how well it works in those languages would be a guess.

So we built a **test with real answers**. NCERT textbooks have exercise questions
at the end of every chapter — we extracted **268 of them** automatically. Then a
human reviews each one and writes the correct answer.

Now we can ask precise questions:

- When a student asks question #47, does the system find the right page?
- Does it find it **first**, or 8th?
- **Is it as good in Marathi as in English?**

That last question is the one this whole project turns on. And the honest
scientific move is to build the measurement *first*, before tuning anything —
because otherwise you can't tell whether a change helped or you just got lucky.

---

## 7. Making the video

The final piece: turn an answer into a **30-second explainer video**.

Four scenes:

| Time | What's on screen | Narration |
|---|---|---|
| 0–4s | Title card | ≤10 words |
| 4–14s | Story scene | ≤25 words |
| 14–25s | Concept diagram | ≤28 words |
| 25–30s | Quiz question | ≤12 words |

**Why the word limits?** Because speech has a speed. People narrate at roughly
2.5 words per second in English, a bit slower in Hindi and Marathi. So 30
seconds ≈ **75 words, total**. Write 150 words and you get a 60-second video —
the limit isn't a style preference, it's arithmetic.

**Why we don't use AI image generation:** We *could* use an AI to draw pictures.
We deliberately don't, for three reasons:

1. AI image models are **terrible at text** — and in an educational video, the
   text *is* the point
2. They give a **different image every time** you run them, which is bad for
   research you have to defend
3. They're slow and need a big download

Instead, diagrams are drawn by **Graphviz** (a proper diagram tool) and text
cards by code. Same input, same video, every single time. Boring is correct here.

**The voice** comes from `indic-parler-tts`, a free model from AI4Bharat and IIT
Madras that speaks Hindi, Marathi and Indian-accented English — all three, in one
model.

---

## 8. Why everything is free

Every single model in this system is **open-source and free**:

| Part | Model | Cost |
|---|---|---|
| Meaning fingerprints | bge-m3 | ₹0 |
| Reranker | bge-reranker-v2-m3 | ₹0 |
| Answer writing | Qwen3 | ₹0 |
| Indian languages | Sarvam-M | ₹0 |
| Voice | indic-parler-tts | ₹0 |
| Translation | IndicTrans2 | ₹0 |

This isn't just about saving money on a student project. It's about whether the
thing can actually exist.

If every question cost even ₹2 to answer, then 1,000 students asking 5 questions
a day would cost **₹3 lakh a month** — forever. No school can pay that. Free
models running on a university's own computers cost **nothing per question**,
which is the difference between a demo and something a real classroom could use.

We run it on **Sol**, ASU's supercomputer, on an NVIDIA A100 — a GPU with 80 GB
of memory, maybe 50 times faster than a laptop for this kind of work. Building
the search index over all 7,992 chunks took **2 minutes 58 seconds**. On a laptop
it took closer to an hour.

---

## 9. The words, in one place

| Term | Plain meaning |
|---|---|
| **RAG** | Look up real sources first, then answer using them |
| **Chunk** | A small piece of a textbook, about a page |
| **Embedding** | A list of numbers representing a text's meaning |
| **Semantic search** | Finding things by meaning, not spelling |
| **BM25** | Finding things by exact word matches |
| **RRF** | Fairly merging two ranked lists using position, not scores |
| **Reranker** | A slow, careful model that double-checks the top few |
| **LLM** | The AI that writes the final answer |
| **Hallucination** | When an AI confidently states something untrue |
| **Citation** | A marker like `[S1]` showing which source a fact came from |
| **GPU** | A processor that does thousands of things at once |
| **vLLM** | Software that serves our AI model efficiently |
| **SLURM** | The supercomputer's queue system for sharing GPUs |

---

## 10. In one paragraph

We took 282 NCERT textbooks in three languages, cut them into 7,992 searchable
pieces, and built a system that finds the right pages two different ways at
once, merges the results fairly, has an expert model double-check the finalists,
and then has an AI write a student-friendly explanation **using only those
pages** — citing each one. It runs entirely on free, open models on a university
supercomputer, so it costs nothing per question. The hard part isn't the
English. It's making it work **equally well in Hindi and Marathi** — and the
honest answer is that we've built the measuring tape but haven't yet read the
measurement.
