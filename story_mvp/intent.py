"""Route an incoming message before spending a retrieval on it.

A curriculum RAG system that answers *every* input by searching textbooks
behaves badly as a chatbot: "hi" retrieves noise, fails the evidence gate, and
comes back with a refusal. That is technically correct and a terrible greeting.

Classification is rule-based rather than model-based on purpose:
  - it must run before retrieval, so it has to be fast and free
  - it must be deterministic, so its behaviour is testable
  - an LLM call here would fail exactly when the LLM is unreachable, which is
    precisely when a graceful greeting matters most

The default is QUESTION. Misrouting a greeting into retrieval is a small
annoyance; misrouting a real question into a canned greeting loses the answer.
"""

from __future__ import annotations

import re
from typing import Dict

GREETING = "greeting"
SMALLTALK = "smalltalk"
CAPABILITY = "capability"
QUESTION = "question"

# Whole-message patterns. Anchored, because "hi" inside "what is high tide"
# is not a greeting.
_GREETING = re.compile(
    r"^(hi|hey|hello|yo|hii+|helo|good\s*(morning|afternoon|evening)|"
    r"नमस्ते|नमस्कार|हाय|हॅलो|हेलो|सलाम)[\s!.,]*$",
    re.IGNORECASE,
)

_SMALLTALK = re.compile(
    r"^(how\s*are\s*you|how'?s\s*it\s*going|thanks?|thank\s*you|thx|ok(ay)?|"
    r"bye|goodbye|see\s*you|nice|cool|great|good|"
    r"धन्यवाद|शुक्रिया|आभारी\s*आहे|ठीक\s*आहे|अच्छा|बरं)[\s!.,]*$",
    re.IGNORECASE,
)

# Asking about the assistant itself, not about the curriculum.
_CAPABILITY = re.compile(
    r"(what\s+(can|do)\s+you\s+(do|help)|who\s+are\s+you|what\s+are\s+you|"
    r"how\s+(do|does)\s+(you|this)\s+work|what\s+is\s+this|help\s*me\s*out|"
    r"^help[\s!.?]*$|which\s+(books?|subjects?|classes?)|"
    r"तुम्ही\s*काय|तुम\s*क्या|आप\s*क्या\s*कर|कोण\s*आहेस)",
    re.IGNORECASE,
)


def classify(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return GREETING

    if _GREETING.match(text):
        return GREETING
    if _SMALLTALK.match(text):
        return SMALLTALK
    if _CAPABILITY.search(text):
        return CAPABILITY

    # A very short message with no question shape is more likely chitchat than
    # a curriculum question worth a retrieval.
    if len(text) < 12 and not text.endswith("?") and " " not in text.strip():
        return SMALLTALK

    return QUESTION


GREETING_REPLY = {
    "english": (
        "Hello! I'm StoryTutor — I answer questions from the NCERT Class 6–8 "
        "Science and Social Science textbooks, in English, Hindi or Marathi.\n\n"
        "Ask me something like *\"Why do we see different phases of the moon?\"* "
        "and I'll answer using the actual textbook pages, with the source shown."
    ),
    "hindi": (
        "नमस्ते! मैं StoryTutor हूँ — मैं NCERT कक्षा 6–8 की विज्ञान और सामाजिक विज्ञान "
        "की किताबों से प्रश्नों के उत्तर देता हूँ, हिंदी, मराठी या अंग्रेज़ी में।\n\n"
        "मुझसे पूछिए, जैसे *\"पौधे भोजन कैसे बनाते हैं?\"* — मैं पाठ्यपुस्तक के पन्नों "
        "से उत्तर दूँगा और स्रोत भी दिखाऊँगा।"
    ),
    "marathi": (
        "नमस्कार! मी StoryTutor — मी NCERT इयत्ता 6–8 च्या विज्ञान आणि सामाजिक शास्त्र "
        "पुस्तकांमधून प्रश्नांची उत्तरे देतो, मराठी, हिंदी किंवा इंग्रजीमध्ये.\n\n"
        "मला विचारा, उदाहरणार्थ *\"वनस्पती अन्न कसे तयार करतात?\"* — मी पाठ्यपुस्तकातील "
        "पानांवरून उत्तर देईन आणि स्रोतही दाखवेन."
    ),
}

CAPABILITY_REPLY = {
    "english": (
        "I answer questions from NCERT textbooks for Class 6, 7 and 8 — Science "
        "and Social Science — in English, Hindi and Marathi.\n\n"
        "Every answer is retrieved from the real textbook pages and cited, so you "
        "can check where it came from. I won't answer from outside those books: if "
        "something isn't in them, I'll say so rather than guess.\n\n"
        "Pick your class and subject below to narrow the search, then ask away."
    ),
    "hindi": (
        "मैं NCERT की कक्षा 6, 7 और 8 की विज्ञान और सामाजिक विज्ञान की पुस्तकों से "
        "प्रश्नों के उत्तर देता हूँ — हिंदी, मराठी और अंग्रेज़ी में।\n\n"
        "हर उत्तर असली पाठ्यपुस्तक के पन्नों से लिया जाता है और स्रोत दिखाया जाता है। "
        "अगर कोई बात उन किताबों में नहीं है, तो मैं अनुमान लगाने के बजाय साफ़ कह दूँगा।"
    ),
    "marathi": (
        "मी NCERT च्या इयत्ता 6, 7 आणि 8 च्या विज्ञान व सामाजिक शास्त्र पुस्तकांमधून "
        "प्रश्नांची उत्तरे देतो — मराठी, हिंदी आणि इंग्रजीमध्ये.\n\n"
        "प्रत्येक उत्तर खऱ्या पाठ्यपुस्तकाच्या पानांवरून घेतले जाते आणि स्रोत दाखवला जातो. "
        "जर एखादी गोष्ट त्या पुस्तकांत नसेल, तर मी अंदाज न लावता तसे स्पष्ट सांगेन."
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
