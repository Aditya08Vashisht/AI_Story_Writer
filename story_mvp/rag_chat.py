"""Clean RAG chat pipeline. No story generation, no genres, no personas.

This replaces the story-generator path for the chat product. The old pipeline
carried vestigial fiction concepts -- genre, tone, "hook/expand/continue" modes,
and a deterministic fallback that wrote thriller prose about locked phones --
which made a curriculum tutor behave like a fiction engine whenever the LLM was
unreachable.

Here the contract is narrow on purpose:
    question + (class, subject, language)  ->  grounded answer + citations

When no model is reachable the fallback does NOT invent prose. It returns the
retrieved textbook passages and says plainly that it could not generate a
summary. A student reading real source text is served; a student reading
invented text is actively misled.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from story_mvp import guardrails
from story_mvp.intent import (
    CAPABILITY,
    GREETING,
    QUESTION,
    SMALLTALK,
    classify,
    conversational_reply,
)

LANGUAGES = {"english", "hindi", "marathi"}
CLASS_LEVELS = {"6", "7", "8"}
SUBJECTS = {"science", "social_science"}

# Below this reranker/retrieval score the evidence is too weak to answer from.
MIN_EVIDENCE_SCORE = 0.25

LANGUAGE_NAMES = {"english": "English", "hindi": "Hindi", "marathi": "Marathi"}


def _clean(value: Any, limit: int = 1000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _choice(value: Any, allowed: set, fallback: str = "") -> str:
    v = _clean(value).lower()
    if v in {"social science", "sst"}:
        v = "social_science"
    return v if v in allowed else fallback


def normalize_chat_request(payload: Dict[str, Any]) -> Dict[str, str]:
    """Accept only what a curriculum question needs."""
    return {
        "question": _clean(payload.get("question") or payload.get("idea") or payload.get("message")),
        "language": _choice(payload.get("language"), LANGUAGES, "english"),
        "class_level": _choice(payload.get("class_level"), CLASS_LEVELS, ""),
        "subject": _choice(payload.get("subject"), SUBJECTS, ""),
    }


def _recent_history(payload: Dict[str, Any], turns: int = 4) -> List[Dict[str, str]]:
    """Last few turns, so follow-ups like 'why?' have something to refer to."""
    raw = payload.get("history") or []
    out = []
    for item in raw[-turns:]:
        role = item.get("role")
        content = _clean(item.get("content"), 600)
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


def build_chat_prompt(request: Dict[str, str], context: str) -> str:
    language = LANGUAGE_NAMES.get(request["language"], "English")
    level = f"Class {request['class_level']}" if request["class_level"] else "Class 6-8"
    subject = request["subject"].replace("_", " ") if request["subject"] else "Science or Social Science"

    return f"""You are a tutor for Indian school students studying the NCERT {subject} curriculum.

Answer the student's question using ONLY the numbered sources below.

Rules:
- Write your answer in {language}.
- Pitch it at {level} level: short sentences, plain words, concrete examples.
- End every factual sentence with its source marker, like [S1] or [S2].
- Use only source numbers that appear below. Never invent a citation.
- If the sources do not contain the answer, say exactly that and name what is
  missing. Do not fill the gap from your own knowledge.
- 2 to 4 short paragraphs. No preamble, no sign-off. Explain the concept
  directly; do not invent people, dialogue or fictional scenarios.

Sources:
---
{context}
---

Return only valid JSON: {{"answer": "your answer here"}}"""


def _source_score(doc: Dict[str, Any]) -> float:
    for key in ("rerank_score", "retrieval_score"):
        value = doc.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _format_sources(results: List[Dict]) -> str:
    parts = []
    for i, doc in enumerate(results, 1):
        bits = [
            f"Class {doc['class_level']}" if doc.get("class_level") else None,
            doc.get("subject"),
            doc.get("language"),
            f"chapter {doc['chapter']}" if doc.get("chapter") else None,
            f"page {doc['page_start']}" if doc.get("page_start") else None,
        ]
        label = ", ".join(b for b in bits if b)
        parts.append(f"[S{i}] ({label})\n{doc.get('text', '')}\n")
    return "\n".join(parts)


def _citations(results: List[Dict]) -> List[Dict[str, Any]]:
    import os

    out = []
    for i, doc in enumerate(results, 1):
        source_file = doc.get("source_file")
        out.append(
            {
                "marker": f"S{i}",
                "id": doc.get("id"),
                "class_level": doc.get("class_level"),
                "subject": doc.get("subject"),
                "language": doc.get("language"),
                "chapter": doc.get("chapter"),
                "page": doc.get("page_start"),
                "source_file": os.path.basename(source_file) if source_file else None,
                "score": _source_score(doc),
                "rerank_score": doc.get("rerank_score"),
                "retrieval_method": doc.get("retrieval_method"),
                "excerpt": _clean(doc.get("text", ""), 320),
            }
        )
    return out


NO_EVIDENCE = {
    "english": "I could not find this in the Class 6-8 NCERT books I have. Try rephrasing, or pick the class and subject so I can search the right book.",
    "hindi": "मुझे यह कक्षा 6-8 की NCERT पुस्तकों में नहीं मिला। कृपया प्रश्न दूसरे शब्दों में पूछें, या कक्षा और विषय चुनें।",
    "marathi": "हे मला इयत्ता 6-8 च्या NCERT पुस्तकांमध्ये सापडले नाही. कृपया प्रश्न वेगळ्या शब्दांत विचारा, किंवा इयत्ता आणि विषय निवडा.",
}

NO_MODEL_NOTE = {
    "english": "No language model is currently reachable, so I cannot summarise. Here are the exact textbook passages that match your question:",
    "hindi": "अभी कोई भाषा मॉडल उपलब्ध नहीं है, इसलिए मैं सारांश नहीं दे सकता। आपके प्रश्न से मेल खाने वाले पाठ्यपुस्तक के अंश नीचे दिए गए हैं:",
    "marathi": "सध्या कोणतेही भाषा मॉडेल उपलब्ध नाही, त्यामुळे मी सारांश देऊ शकत नाही. तुमच्या प्रश्नाशी जुळणारे पाठ्यपुस्तकातील उतारे खाली दिले आहेत:",
}


def _passages_fallback(request: Dict[str, str], results: List[Dict]) -> str:
    """Honest degradation: show the real sources, never invent prose."""
    lang = request["language"]
    lines = [NO_MODEL_NOTE.get(lang, NO_MODEL_NOTE["english"]), ""]
    for i, doc in enumerate(results, 1):
        where = []
        if doc.get("chapter"):
            where.append(f"chapter {doc['chapter']}")
        if doc.get("page_start"):
            where.append(f"page {doc['page_start']}")
        lines.append(f"[S{i}] {', '.join(where)}")
        lines.append(_clean(doc.get("text", ""), 600))
        lines.append("")
    return "\n".join(lines).strip()


def answer_question(
    payload: Dict[str, Any],
    rag_engine,
    llm_client=None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Question in, grounded answer + citations out."""
    request = normalize_chat_request(payload)
    history = _recent_history(payload)

    if not request["question"]:
        return {
            "answer": conversational_reply(GREETING, request["language"]),
            "sources": [], "grounded": False, "model_provider": "conversational",
            "retrieval_method": None, "intent": GREETING,
            **request,
        }

    # --- guardrail: input ---
    allowed, reason, guard_meta = guardrails.check_input(request["question"])
    if not allowed:
        return {
            "answer": guardrails.refusal_message(reason, request["language"]),
            "sources": [], "grounded": False, "model_provider": "guardrail",
            "retrieval_method": None, "intent": "blocked", "blocked_reason": reason,
            **request,
        }
    if guard_meta["pii_types"]:
        request["question"] = guardrails.redact_pii(request["question"])

    # --- route: not every message deserves a retrieval ---
    intent = classify(request["question"])
    if intent != QUESTION:
        return {
            "answer": conversational_reply(intent, request["language"]),
            "sources": [], "grounded": True, "model_provider": "conversational",
            "retrieval_method": None, "intent": intent,
            **request,
        }

    filters = {
        "class_level": request["class_level"],
        "subject": request["subject"],
        "language": request["language"],
    }
    results = rag_engine.retrieve(query=request["question"], filters=filters, top_k=top_k)

    strong = [doc for doc in results if _source_score(doc) >= MIN_EVIDENCE_SCORE]
    retrieval_method = results[0].get("retrieval_method") if results else None

    # Evidence gate: refuse rather than guess when retrieval found nothing usable.
    if not strong:
        return {
            "answer": NO_EVIDENCE.get(request["language"], NO_EVIDENCE["english"]),
            "sources": _citations(results),
            "grounded": False,
            "model_provider": "none",
            "retrieval_method": retrieval_method,
            "intent": QUESTION,
            **request,
        }

    context = _format_sources(strong)

    if llm_client is not None:
        try:
            answer = _generate(llm_client, request, context, history)
            if answer:
                checked = guardrails.check_output(answer, strong)
                return {
                    "answer": checked["answer"],
                    "sources": _citations(strong),
                    "grounded": True,
                    "model_provider": getattr(llm_client, "last_provider", "unknown"),
                    "retrieval_method": retrieval_method,
                    "intent": QUESTION,
                    "groundedness": checked["groundedness"],
                    "low_groundedness": checked["low_groundedness"],
                    "citations_used": checked["citations_used"],
                    "invented_citations": checked["invented_citations"],
                    **request,
                }
        except Exception as exc:  # noqa: BLE001 - any model failure degrades the same way
            print(f"chat: generation failed, returning passages instead - {exc}")

    return {
        "answer": _passages_fallback(request, strong),
        "sources": _citations(strong),
        "grounded": True,
        "model_provider": "passages_only",
        "retrieval_method": retrieval_method,
        "intent": QUESTION,
        **request,
    }


def _generate(llm_client, request: Dict[str, str], context: str,
              history: List[Dict[str, str]] = None) -> Optional[str]:
    """Call the provider chain with a chat-shaped prompt."""
    from story_mvp.model_clients import parse_json_response

    system = build_chat_prompt(request, context)
    user = request["question"]

    client = getattr(llm_client, "client", llm_client)
    for provider in getattr(client, "clients", [client]):
        try:
            raw = _call_provider(provider, system, user, history)
        except Exception as exc:  # noqa: BLE001
            print(f"chat: {getattr(provider, 'provider_name', '?')} failed - {exc}")
            continue
        try:
            answer = parse_json_response(raw).get("answer", "").strip()
        except Exception:
            answer = _clean(raw, 4000)
        if answer:
            llm_client.last_provider = getattr(provider, "provider_name", "unknown")
            if hasattr(client, "last_provider"):
                client.last_provider = llm_client.last_provider
            return answer
    return None


def _call_provider(provider, system: str, user: str,
                   history: List[Dict[str, str]] = None) -> str:
    """Send one chat completion. Providers differ only in transport shape."""
    import json
    import urllib.request

    name = getattr(provider, "provider_name", "")
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user})

    if name == "groq":
        completion = provider.client.chat.completions.create(
            messages=messages,
            model=provider.model,
            temperature=0.3,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    if name == "ollama":
        body = {"model": provider.model, "stream": False, "format": "json",
                "messages": messages, "options": {"temperature": 0.3}}
        req = urllib.request.Request(
            f"{provider.host}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=getattr(provider, "timeout", 180)) as r:
            return json.loads(r.read().decode("utf-8")).get("message", {}).get("content", "")

    # vllm and sarvam are both OpenAI-compatible
    base = getattr(provider, "base_url", None) or getattr(provider, "endpoint", "")
    url = f"{base}/chat/completions" if base and not base.endswith("/chat/completions") else base
    body = {"model": provider.model, "messages": messages, "temperature": 0.3, "max_tokens": 1200}
    if name == "vllm":
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    key = getattr(provider, "api_key", None)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=getattr(provider, "timeout", 180)) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "")
