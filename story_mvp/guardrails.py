"""Input and output guardrails for the curriculum chatbot.

Deliberately rule-based. A model-based guard fails exactly when the model is
unreachable, which is when protection matters most, and it cannot be unit
tested deterministically. These checks are cheap, always available, and each
one is verifiable.

The users are 11-14 year olds, so two things matter more than they would in a
general assistant: never fabricate a citation (a wrong source is worse than no
source, because it looks checkable), and never silently log personal data.

plan.md Sec.E lists the model-based upgrades (HHEM groundedness, Granite
Guardian) that layer on top of this.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

MAX_QUESTION_CHARS = 1200

# Attempts to talk past the system prompt rather than ask a question.
_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)|"
    r"disregard\s+(the\s+)?(system|above|previous)|"
    r"you\s+are\s+now\s+(a|an)\s|"
    r"pretend\s+(to\s+be|you\s+are)|"
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)|"
    r"output\s+your\s+(prompt|instructions)|"
    r"</?(system|instruction)>)",
    re.IGNORECASE,
)

# Indian phone numbers, email, and Aadhaar-shaped 12-digit runs.
_PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("id_number", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
]

_CITATION = re.compile(r"\[S(\d+)\]")


def check_input(question: str) -> Tuple[bool, str, Dict]:
    """Returns (allowed, reason, metadata). Never raises."""
    text = (question or "").strip()
    meta: Dict = {"length": len(text), "pii_types": []}

    if len(text) > MAX_QUESTION_CHARS:
        return False, "too_long", meta

    if _INJECTION.search(text):
        return False, "prompt_injection", meta

    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            meta["pii_types"].append(label)

    # PII is redacted rather than refused: a student typing their phone number
    # into a question about the postal system should still get an answer.
    return True, "ok", meta


def redact_pii(text: str) -> str:
    out = text or ""
    for label, pattern in _PII_PATTERNS:
        out = pattern.sub(f"[{label} removed]", out)
    return out


REFUSAL = {
    "prompt_injection": {
        "english": "I can only answer questions from the NCERT Class 6-8 textbooks. Ask me something from your syllabus.",
        "hindi": "मैं केवल NCERT कक्षा 6-8 की पुस्तकों से प्रश्नों के उत्तर दे सकता हूँ। अपने पाठ्यक्रम से कुछ पूछिए।",
        "marathi": "मी फक्त NCERT इयत्ता 6-8 च्या पुस्तकांमधील प्रश्नांची उत्तरे देऊ शकतो. तुमच्या अभ्यासक्रमातून काहीतरी विचारा.",
    },
    "too_long": {
        "english": "That question is too long. Try asking it in a sentence or two.",
        "hindi": "यह प्रश्न बहुत लंबा है। इसे एक-दो वाक्यों में पूछिए।",
        "marathi": "हा प्रश्न खूप मोठा आहे. तो एक-दोन वाक्यांत विचारा.",
    },
}


def refusal_message(reason: str, language: str) -> str:
    table = REFUSAL.get(reason, REFUSAL["prompt_injection"])
    return table.get(language, table["english"])


def validate_citations(answer: str, n_sources: int) -> Tuple[str, Dict]:
    """Strip citation markers that point at sources which do not exist.

    A model asked for [S1]-style markers will occasionally emit [S7] when only
    five sources were supplied. Leaving it in is worse than having no citation
    at all: it looks verifiable, and a student who tries to check it finds
    nothing. So invented markers are removed and counted.
    """
    invented: List[str] = []

    def _keep(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            return match.group(0)
        invented.append(match.group(0))
        return ""

    cleaned = _CITATION.sub(_keep, answer or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    used = sorted({int(m) for m in _CITATION.findall(cleaned)})
    return cleaned.strip(), {
        "citations_used": used,
        "citation_count": len(used),
        "invented_citations": invented,
        "has_citations": bool(used),
    }


def _tokens(text: str) -> set:
    return set(re.findall(r"[\wऀ-ॣ०-ॿ]{4,}", (text or "").lower()))


def groundedness(answer: str, sources: List[Dict]) -> float:
    """Cheap lexical proxy: what fraction of the answer's content words appear
    in the retrieved text.

    This is not entailment. It catches an answer that drifted entirely off the
    sources, which is the failure mode that matters here; it will not catch a
    subtle factual inversion. plan.md Sec.E schedules HHEM-2.1-open for that.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    source_tokens = set()
    for s in sources:
        source_tokens |= _tokens(s.get("text") or s.get("excerpt") or "")
    if not source_tokens:
        return 0.0
    return round(len(answer_tokens & source_tokens) / len(answer_tokens), 3)


# Below this, the answer shares almost no vocabulary with its sources.
LOW_GROUNDEDNESS = 0.25


def check_output(answer: str, sources: List[Dict]) -> Dict:
    cleaned, citation_meta = validate_citations(answer, len(sources))
    score = groundedness(cleaned, sources)
    return {
        "answer": cleaned,
        "groundedness": score,
        "low_groundedness": score < LOW_GROUNDEDNESS,
        **citation_meta,
    }
