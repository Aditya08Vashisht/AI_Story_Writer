"""Intent understanding, in three tiers of decreasing intelligence.

A curriculum RAG system that sends every input to retrieval replies to "hi"
with a refusal. But the fix should not be a pattern list: regexes cannot
handle "yo whats good", typos, unseen phrasings, code-mixed Hinglish, or
mixed intent like "hi, explain photosynthesis".

So routing is layered, best first:

  1. THE MODEL DECIDES (primary). One prompt covers greeting, small talk,
     capability questions and grounded answering. Retrieval always runs -- it
     costs ~50 ms -- and the model judges whether the sources are relevant.
     This is how a real assistant behaves and it needs no classifier at all.

  2. SEMANTIC (fallback). When no model is reachable, compare the message to
     prototype phrases in embedding space, using the bge-m3 model already
     loaded for retrieval. Nearly free, and genuinely handles paraphrase,
     typos and languages the prototypes were never written in.

  3. LEXICAL (last resort). Only when neither of the above is available.

Guardrails are deliberately NOT part of this. A security check that depends on
model judgement can be argued out of its decision; those stay deterministic.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

GREETING = "greeting"
SMALLTALK = "smalltalk"
CAPABILITY = "capability"
QUESTION = "question"

# Prototype utterances per intent. These are embedded once and compared by
# cosine similarity, so they act as semantic anchors rather than patterns --
# "sup", "kaise ho" and "namaskar re" all land near the greeting cluster
# without ever being listed.
PROTOTYPES: Dict[str, List[str]] = {
    GREETING: [
        "hi", "hello", "hey there", "good morning", "good evening",
        "नमस्ते", "नमस्कार", "हाय", "कसे आहात", "कैसे हो",
    ],
    SMALLTALK: [
        "thanks a lot", "thank you, that helped", "okay got it", "bye",
        "nice one", "cool thanks", "धन्यवाद", "ठीक आहे", "अच्छा समझ गया",
    ],
    CAPABILITY: [
        "what can you do", "who are you", "how does this work",
        "which subjects do you cover", "what books do you have", "help me",
        "तुम्ही काय करू शकता", "आप क्या कर सकते हैं",
    ],
    QUESTION: [
        "why does ice float on water",
        "explain photosynthesis in plants",
        "what caused the revolt of 1857",
        "how do magnets attract iron",
        "पौधे भोजन कैसे बनाते हैं",
        "वनस्पती अन्न कसे तयार करतात",
        "भारत के प्रमुख संसाधन कौन से हैं",
    ],
}

# Below this cosine similarity to every prototype, the message looks like
# nothing we anchored -- which for an open-ended tutor means a real question.
_MIN_SIMILARITY = 0.45


class SemanticIntent:
    """Embedding-space intent, reusing the retrieval model already in memory."""

    def __init__(self, embed_model):
        self.model = embed_model
        self._labels: List[str] = []
        self._matrix = None
        self._ready = False

    def _ensure(self) -> bool:
        if self._ready:
            return self._matrix is not None
        self._ready = True
        try:
            import numpy as np

            texts, labels = [], []
            for intent, phrases in PROTOTYPES.items():
                texts.extend(phrases)
                labels.extend([intent] * len(phrases))
            vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self._matrix = np.asarray(vecs, dtype="float32")
            self._labels = labels
        except Exception as exc:  # noqa: BLE001
            print(f"intent: semantic classifier unavailable, using lexical - {exc}")
            self._matrix = None
        return self._matrix is not None

    def classify(self, message: str) -> Tuple[str, float]:
        if not self._ensure():
            return classify_lexical(message), 0.0
        import numpy as np

        q = self.model.encode([message], convert_to_numpy=True, normalize_embeddings=True)
        sims = np.asarray(q, dtype="float32") @ self._matrix.T
        best = int(sims[0].argmax())
        score = float(sims[0][best])
        if score < _MIN_SIMILARITY:
            return QUESTION, score
        return self._labels[best], score


# ---------------------------------------------------------------- tier 3

_GREETING = re.compile(
    r"^(hi+|hey+|hello+|yo|helo|sup|good\s*(morning|afternoon|evening)|"
    r"नमस्ते|नमस्कार|हाय|हॅलो|हेलो)[\s!.,]*$",
    re.IGNORECASE,
)
_SMALLTALK = re.compile(
    r"^(how\s*are\s*you|thanks?|thank\s*you|thx|ty|ok(ay)?|bye|goodbye|"
    r"nice|cool|great|got\s*it|धन्यवाद|शुक्रिया|ठीक\s*आहे|अच्छा)[\s!.,]*$",
    re.IGNORECASE,
)
_CAPABILITY = re.compile(
    r"(what\s+(can|do)\s+you\s+(do|help)|who\s+are\s+you|what\s+are\s+you|"
    r"how\s+(do|does)\s+(you|this)\s+work|^help[\s!.?]*$|"
    r"which\s+(books?|subjects?|classes?)|तुम्ही\s*काय|आप\s*क्या\s*कर)",
    re.IGNORECASE,
)


def classify_lexical(message: str) -> str:
    """Last-resort routing. Defaults to QUESTION: misrouting a greeting is an
    annoyance, misrouting a question loses the answer."""
    text = (message or "").strip()
    if not text:
        return GREETING
    if _GREETING.match(text):
        return GREETING
    if _SMALLTALK.match(text):
        return SMALLTALK
    if _CAPABILITY.search(text):
        return CAPABILITY
    if len(text) < 12 and not text.endswith("?") and " " not in text:
        return SMALLTALK
    return QUESTION


# Kept as the public name so callers need not know which tier answered.
classify = classify_lexical


# ------------------------------------------------- canned replies (tier 2/3)

GREETING_REPLY = {
    "english": (
        "Hello! I'm StoryTutor — I answer questions from the NCERT Class 6–8 "
        "Science and Social Science textbooks, in English, Hindi or Marathi.\n\n"
        "Try asking something like \"Why do we see different phases of the moon?\" "
        "and I'll answer from the actual textbook pages, showing you the source."
    ),
    "hindi": (
        "नमस्ते! मैं StoryTutor हूँ — मैं NCERT कक्षा 6–8 की विज्ञान और सामाजिक विज्ञान "
        "की किताबों से प्रश्नों के उत्तर देता हूँ, हिंदी, मराठी या अंग्रेज़ी में।\n\n"
        "मुझसे पूछिए, जैसे \"पौधे भोजन कैसे बनाते हैं?\" — मैं पाठ्यपुस्तक के पन्नों से "
        "उत्तर दूँगा और स्रोत भी दिखाऊँगा।"
    ),
    "marathi": (
        "नमस्कार! मी StoryTutor — मी NCERT इयत्ता 6–8 च्या विज्ञान आणि सामाजिक शास्त्र "
        "पुस्तकांमधून प्रश्नांची उत्तरे देतो, मराठी, हिंदी किंवा इंग्रजीमध्ये.\n\n"
        "मला विचारा, उदाहरणार्थ \"वनस्पती अन्न कसे तयार करतात?\" — मी पाठ्यपुस्तकातील "
        "पानांवरून उत्तर देईन आणि स्रोतही दाखवेन."
    ),
}

CAPABILITY_REPLY = {
    "english": (
        "I answer questions from NCERT textbooks for Class 6, 7 and 8 — Science and "
        "Social Science — in English, Hindi and Marathi.\n\n"
        "Every answer is retrieved from the real textbook pages and cited, so you can "
        "check where it came from. I won't answer from outside those books: if "
        "something isn't in them, I'll say so rather than guess."
    ),
    "hindi": (
        "मैं NCERT की कक्षा 6, 7 और 8 की विज्ञान और सामाजिक विज्ञान की पुस्तकों से "
        "प्रश्नों के उत्तर देता हूँ — हिंदी, मराठी और अंग्रेज़ी में।\n\n"
        "हर उत्तर असली पाठ्यपुस्तक के पन्नों से लिया जाता है और स्रोत दिखाया जाता है।"
    ),
    "marathi": (
        "मी NCERT च्या इयत्ता 6, 7 आणि 8 च्या विज्ञान व सामाजिक शास्त्र पुस्तकांमधून "
        "प्रश्नांची उत्तरे देतो — मराठी, हिंदी आणि इंग्रजीमध्ये.\n\n"
        "प्रत्येक उत्तर खऱ्या पाठ्यपुस्तकाच्या पानांवरून घेतले जाते आणि स्रोत दाखवला जातो."
    ),
}

SMALLTALK_REPLY = {
    "english": "Happy to help. Ask me anything from your Class 6–8 Science or Social Science textbook.",
    "hindi": "मुझे मदद करके खुशी होगी। कक्षा 6–8 की विज्ञान या सामाजिक विज्ञान की किताब से कुछ भी पूछिए।",
    "marathi": "मदत करायला आनंद होईल. इयत्ता 6–8 च्या विज्ञान किंवा सामाजिक शास्त्र पुस्तकातून काहीही विचारा.",
}

_REPLIES: Dict[str, Dict[str, str]] = {
    GREETING: GREETING_REPLY,
    CAPABILITY: CAPABILITY_REPLY,
    SMALLTALK: SMALLTALK_REPLY,
}


def conversational_reply(intent: str, language: str) -> str:
    table = _REPLIES.get(intent, SMALLTALK_REPLY)
    return table.get(language, table["english"])
