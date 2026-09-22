"""Apply the first human(-proxy) review pass to review_sample_50.jsonl.

This encodes a manual, meaning-level review of all 50 sampled questions:
read each one in full chunk context, judged for (a) whether it's a real,
answerable standalone question and (b) whether the NCERT-correct answer
could be written confidently. Reference answers are written in English
regardless of question language, since the eval harness grades semantic
content, not language match.

IMPORTANT CAVEAT: this review was done by an LLM acting as a review proxy,
not a Marathi/Hindi-fluent human. Confidence is noted per item. Per
plan.md Sec.8.1, a fluent bilingual human pass -- especially over the
"keep_noisy" Marathi items, where OCR corruption is worst -- should still
happen before these numbers are treated as fully trustworthy.

status values:
  keep_clean  - legible, standalone, answerable with confidence
  keep_noisy  - meaning recovered despite OCR noise; text could use cleanup
  reject      - not usable: corrupted beyond confident reconstruction,
                not actually a question (caption/credits/meta-text), or
                depends on external content (image/table/prior page) not
                present in the chunk
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REVIEW = {
    "golden_0256": dict(status="reject", reason="severely corrupted OCR (broken conjuncts), low-confidence reconstruction"),
    "golden_0262": dict(status="reject", reason="severely corrupted OCR"),
    "golden_0105": dict(status="reject", reason="mid-sentence quoted fragment, not a standalone question"),
    "golden_0234": dict(status="reject", reason="severely corrupted OCR"),
    "golden_0006": dict(status="keep_clean", question_type="conceptual",
        reference_answer="Natural grass cools its surroundings through evaporation/transpiration and reflects more heat; plastic grass has no moisture, so it absorbs and re-radiates more heat, making the surrounding air warmer."),
    "golden_0226": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="With two moons, Earth would likely have more complex and stronger tidal patterns, more frequent eclipses, and nights that are sometimes brighter when both moons are visible."),
    "golden_0250": dict(status="keep_clean", question_type="open_ended",
        reference_answer=None, note="Open-ended creative-writing prompt; grade for historically accurate facts about the chosen Maratha leader and coherent structure, not a fixed answer."),
    "golden_0200": dict(status="keep_noisy", question_type="factual",
        reference_answer="Stay healthy by getting enough sleep, eating a balanced diet, exercising regularly, spending time with family/friends, and keeping a positive attitude; avoid excessive screen time, junk food, substance use, and chronic stress.",
        note="Leading running-header text ('30 Curiosity...') should be stripped before use."),
    "golden_0010": dict(status="keep_noisy", question_type="factual",
        reference_answer="Delhi-Lucknow distance -> kilometre; coin thickness -> millimetre; eraser length -> centimetre; school ground length -> metre.",
        note="Matching-table format (Column I/Column II); needs a table UI, not plain free-text grading."),
    "golden_0237": dict(status="keep_clean", question_type="open_ended",
        reference_answer=None, note="Analytical/open-ended; grade for use of chapter evidence about the extent of Maratha territory the British later absorbed."),
    "golden_0212": dict(status="reject", reason="corrupted + repeated page-footer stamp text swallowing the actual question"),
    "golden_0052": dict(status="keep_clean", question_type="factual",
        reference_answer="Central ideas of Buddhism include the Four Noble Truths, the Eightfold Path, karma and rebirth, and achieving nirvana through right conduct, meditation, and wisdom."),
    "golden_0268": dict(status="keep_noisy", question_type="application",
        reference_answer=None, note="State-dependent research question; direct the student to look up Lok Sabha and Rajya Sabha seat counts for their own state."),
    "golden_0239": dict(status="keep_clean", question_type="application",
        reference_answer=None, note="Same as golden_0268 (English twin) -- state-dependent."),
    "golden_0185": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="Markets let people exchange goods and services, meet daily needs, provide livelihoods, and connect producers with consumers, supporting a community's economic activity."),
    "golden_0191": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="Climate shapes which crops are grown and what economic activities are viable, influences cultural practices (festivals, food, clothing, housing), and affects social patterns such as migration and settlement."),
    "golden_0227": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="No -- early Earth's atmosphere had little free oxygen; today's oxygen-rich atmosphere developed over billions of years mainly through photosynthesis by early life, and more recently human activity has further changed its composition (added CO2, pollutants)."),
    "golden_0265": dict(status="reject", reason="not a question -- credits/acknowledgment list"),
    "golden_0201": dict(status="reject", reason="corrupted + repeated page-footer stamp text"),
    "golden_0252": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="The judiciary uses judicial review to strike down or correct any law or government action that violates the Constitution, upholding constitutional supremacy and protecting citizens' rights."),
    "golden_0203": dict(status="keep_noisy", question_type="experiment",
        reference_answer="The pencil appears bent or displaced at the water's surface due to refraction: light bends as it passes between water and air, which have different optical densities."),
    "golden_0123": dict(status="keep_noisy", question_type="experiment",
        reference_answer="The balloon bursts with a loud pop because the pin punctures the stretched rubber surface, and the compressed air inside escapes rapidly through the puncture."),
    "golden_0243": dict(status="keep_clean", question_type="conceptual",
        reference_answer="Human capital (workers' education, skills, health, and training) raises productivity; its facilitators include education and training, healthcare, nutrition, and on-the-job experience."),
    "golden_0074": dict(status="reject", reason="references an unlisted set of images ('these') with no textual antecedent in the chunk"),
    "golden_0067": dict(status="keep_noisy", question_type="factual",
        reference_answer="Local time is based on the sun's position at a specific longitude (15 degrees of longitude = 1 hour difference); standard time is a single fixed time for an entire country/region, adopted for convenience.",
        note="Trailing PDF-footer/timestamp junk should be stripped."),
    "golden_0030": dict(status="keep_noisy", question_type="experiment",
        reference_answer="Generally yes -- magnetic attraction depends on the magnet's poles, not its shape, though the visible pattern of attracted filings/pins can differ with a differently shaped magnet."),
    "golden_0124": dict(status="reject", reason="heavy OCR corruption plus references a table (2.2) not present in the chunk"),
    "golden_0066": dict(status="keep_clean", question_type="factual",
        reference_answer="A civilisation is an advanced stage of social development marked by cities, organized government, writing, social classes, art, and technology (e.g. Indus Valley, Mesopotamian, Egyptian civilisations)."),
    "golden_0219": dict(status="reject", reason="severely corrupted OCR, low-confidence reconstruction"),
    "golden_0080": dict(status="reject", reason="fragment referencing 'these two pictures', not present in the extracted text"),
    "golden_0147": dict(status="keep_clean", question_type="context_dependent",
        reference_answer=None, note="Answer depends on which historical period the chapter names; grade against retrieved chapter context, not a fixed reference."),
    "golden_0004": dict(status="keep_clean", question_type="conceptual",
        reference_answer="Depends on the illustration's fuel (commonly LPG or electricity). Example for LPG: chemical energy converted to heat; benefit = quick, controllable, clean-burning; drawback = non-renewable fossil fuel, produces CO2.",
        note="Illustration-dependent; general example given."),
    "golden_0133": dict(status="keep_noisy", question_type="factual",
        reference_answer="Mainly in the leaves (chloroplast-rich mesophyll cells), since leaves are broad, thin, and chlorophyll-rich; some photosynthesis can also occur in green stems.",
        note="Medium-confidence reconstruction."),
    "golden_0182": dict(status="keep_noisy", question_type="conceptual",
        reference_answer="A market's main features: buyers and sellers exchanging goods/services for money, price set by supply and demand, and multiple participants facilitating trade."),
    "golden_0242": dict(status="keep_clean", question_type="factual",
        reference_answer="Direct election example: Lok Sabha/state Assembly elections, where citizens vote directly. Indirect election example: President of India or Rajya Sabha, elected by elected representatives rather than directly by citizens."),
    "golden_0266": dict(status="reject", reason="incomplete, truncated mid-sentence fragment"),
    "golden_0246": dict(status="keep_clean", question_type="conceptual",
        reference_answer="A just and harmonious society ensures fairness and equal treatment, reduces conflict and discrimination, promotes cooperation and stability, and lets people live with dignity and mutual respect."),
    "golden_0245": dict(status="reject", reason="figure caption, not a question"),
    "golden_0122": dict(status="reject", reason="meta-text about how to ask creative questions, not itself an exercise question"),
    "golden_0222": dict(status="keep_noisy", question_type="experiment",
        reference_answer="'Puri' here means the fried bread, not a bridge. Uneven oil contact, heat, or pressing technique on each side during frying makes one side puff/expand more (thinner-walled) than the other.",
        note="Corrected during review: chunk context ('Curiosity textbook Class 8', 'let's return to the question from page 1') clarifies 'puri' = poori, not misread as 'bridge'."),
    "golden_0005": dict(status="keep_clean", question_type="application",
        reference_answer="Plant and protect trees and native plants, join community tree-planting drives, water young saplings, avoid artificial turf, and encourage family/school/neighbours to participate."),
    "golden_0079": dict(status="reject", reason="references undefined 'this/these' with no textual antecedent (likely an image)"),
    "golden_0258": dict(status="reject", reason="not a question -- credits list"),
    "golden_0197": dict(status="reject", reason="severely corrupted, appears to merge fragments of unrelated sentences"),
    "golden_0062": dict(status="keep_clean", question_type="factual",
        reference_answer="Oceans: Pacific, Atlantic, Indian, Southern, Arctic. Continents: Asia, Africa, North America, South America, Antarctica, Europe, Australia/Oceania. Land is concentrated in the Northern Hemisphere; oceans dominate the Southern Hemisphere."),
    "golden_0086": dict(status="reject", reason="figure caption, not a question"),
    "golden_0135": dict(status="reject", reason="not a question -- credits line"),
    "golden_0202": dict(status="keep_noisy", question_type="experiment",
        reference_answer="No -- a pond typically hosts multiple fish and other organisms of varying species and sizes, reflecting biodiversity rather than a single species."),
    "golden_0153": dict(status="keep_clean", question_type="factual",
        reference_answer="Indian agriculture is monsoon-dependent, dominated by small/fragmented landholdings, largely labour-intensive (with rising mechanization), mixes subsistence and commercial farming, and grows diverse crops (rice, wheat, cotton, sugarcane) due to varied climate and soil."),
    "golden_0043": dict(status="reject", reason="fragment referencing an earlier page's question, plus corrupted text"),
}


def main() -> int:
    src = Path("eval/golden/review_sample_50.jsonl")
    reviewed_out = Path("eval/golden/review_sample_50_reviewed.jsonl")
    golden_path = Path("eval/golden/golden_set.jsonl")
    golden_out = Path("eval/golden/golden_set_verified.jsonl")

    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    missing = [r["id"] for r in records if r["id"] not in REVIEW]
    if missing:
        print(f"WARNING: {len(missing)} ids have no review entry: {missing}")

    counts = Counter()
    lang_counts = defaultdict(lambda: Counter())
    reviewed = []
    for r in records:
        verdict = REVIEW.get(r["id"])
        if verdict is None:
            reviewed.append(r)
            continue
        r = dict(r)
        r["status"] = verdict["status"]
        r["needs_review"] = False
        r["reviewed_by"] = "llm_proxy_v1"
        if verdict["status"] == "reject":
            r["reject_reason"] = verdict["reason"]
        else:
            r["question_type"] = verdict.get("question_type", "unclassified")
            r["reference_answer"] = verdict.get("reference_answer")
            if verdict.get("note"):
                r["review_note"] = verdict["note"]
        counts[verdict["status"]] += 1
        lang_counts[r["language"]][verdict["status"]] += 1
        reviewed.append(r)

    with reviewed_out.open("w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(reviewed)} reviewed records -> {reviewed_out}")

    # Propagate verdicts back into the full golden set (268 records): mark the
    # 50 reviewed ones as verified/rejected, leave the other 218 untouched
    # (still needs_review=true) since only the sample of 50 has been checked.
    golden_records = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    review_by_id = {r["id"]: r for r in reviewed if r["id"] in REVIEW}
    for g in golden_records:
        if g["id"] in review_by_id:
            rr = review_by_id[g["id"]]
            g["status"] = rr["status"]
            g["needs_review"] = False
            g["reviewed_by"] = "llm_proxy_v1"
            if rr["status"] == "reject":
                g["reject_reason"] = rr["reject_reason"]
            else:
                g["question_type"] = rr["question_type"]
                g["reference_answer"] = rr["reference_answer"]
                if "review_note" in rr:
                    g["review_note"] = rr["review_note"]

    with golden_out.open("w", encoding="utf-8") as f:
        for g in golden_records:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"wrote {len(golden_records)} records ({len(review_by_id)} verified) -> {golden_out}")

    print(f"\n--- review summary (n={sum(counts.values())}) ---")
    for status in ("keep_clean", "keep_noisy", "reject"):
        print(f"  {status}: {counts[status]}")
    kept = counts["keep_clean"] + counts["keep_noisy"]
    print(f"  TOTAL KEEP: {kept}/{sum(counts.values())} ({100*kept/sum(counts.values()):.0f}%)")

    print("\n--- keep-rate by language ---")
    for lang, c in sorted(lang_counts.items()):
        total = sum(c.values())
        kept = c["keep_clean"] + c["keep_noisy"]
        print(f"  {lang}: {kept}/{total} kept ({100*kept/total:.0f}%)  [clean={c['keep_clean']} noisy={c['keep_noisy']} reject={c['reject']}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
