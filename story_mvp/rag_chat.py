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
    classify_lexical,
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
    """One prompt for every message, written to teach rather than paraphrase.

    Two design decisions carry most of the weight here.

    First, no classifier runs before this. The model can see whether the
    student is greeting it, asking what it does, or asking a curriculum
    question -- and it handles mixed intent ("hi, explain photosynthesis")
    that no router splits cleanly. Retrieval runs either way; irrelevant
    passages can simply be ignored.

    Second, facts and pedagogy are separated. The earlier version demanded a
    citation on every sentence, which is safe and produces a citation-studded
    paraphrase of the textbook -- exactly what the student already failed to
    understand. A retrieved fact needs [S1]. An analogy does not, because an
    analogy is not a factual claim; it only has to avoid contradicting the
    sources. That distinction is what lets the model actually explain.
    """
    language = LANGUAGE_NAMES.get(request["language"], "English")
    class_level = request["class_level"]
    subject = request["subject"].replace("_", " ") if request["subject"] else "Science and Social Science"

    if class_level == "6":
        pitch = ("Class 6 (about 11 years old). Short sentences. Everyday words only. "
                 "Anchor every idea to something they can see or touch.")
    elif class_level == "8":
        pitch = ("Class 8 (about 13 years old). They can handle a mechanism and a "
                 "technical term, as long as you define it the first time.")
    else:
        pitch = ("Class 6-8 (11-13 years old). Assume curiosity, not prior "
                 "vocabulary. Define any technical term you use.")

    sources_block = context.strip() or "(no relevant textbook passages were found)"

    return f"""You are StoryTutor, a warm and genuinely good tutor for Indian school
students studying the NCERT {subject} curriculum.

Reply in {language}. Pitch: {pitch}

Below are textbook passages retrieved for whatever the student just said. They
may or may not be relevant -- judge that yourself.

HOW TO RESPOND

If they are greeting you, thanking you, or making small talk:
  Reply naturally and briefly, then invite a question from their textbook.
  Ignore the passages entirely.

If they ask what you are or what you can do:
  Explain that you answer from the NCERT class 6-8 Science and Social Science
  books in English, Hindi and Marathi, always showing which page it came from.

If they ask a curriculum question the passages DO cover -- explain it properly:
  1. Answer the actual question directly, in one or two sentences.
  2. Explain the mechanism -- the why underneath the what.
  3. Give an intuition or analogy that makes it click.
  4. Give one concrete everyday example an Indian student would recognise
     (chai, monsoon, bicycle, cricket, cooking, the local market).
  5. Optionally end with one short question that checks they followed.

If they ask a curriculum question the passages do NOT cover:
  Say so plainly, name what seems to be missing, and suggest they pick the
  right class and subject. Do not answer from your own knowledge.

FACTS VERSUS EXPLANATION -- this distinction matters

  A FACT about the curriculum must come from the passages and must end with
  its source marker: "Heat moves from the hotter object to the cooler one [S1]."

  Your ANALOGIES, intuitions, everyday examples and framing are your own work.
  They need no citation. They must not contradict the passages, but you should
  absolutely use them -- an explanation without one is just the textbook again,
  and the student already did not understand the textbook.

  So this is right:
    "Heat always flows from hotter to cooler, never the other way [S1].
     Think of it like water finding its level -- it only runs downhill.
     That is why the steel spoon in your chai gets hot but the plastic
     handle stays cool enough to hold."
  The first sentence is cited because it is a curriculum fact. The analogy and
  the chai example are yours, and they are what makes it teach.

NEVER
  - invent a citation, or use a source number not listed below
  - state a curriculum fact the passages do not support
  - answer an out-of-syllabus question from general knowledge
  - invent characters, dialogue or fictional scenes

Keep it to 3-5 short paragraphs. Write like a good teacher talking to one
student, not like a textbook.

TEXTBOOK PASSAGES
---
{sources_block}
---

Return only valid JSON: {{"answer": "your reply here"}}"""


def build_no_source_prompt(request: Dict[str, str]) -> str:
    """Prompt used when retrieval returned nothing usable.

    The full prompt tells the model not to answer from its own knowledge, and
    a 7B model does not reliably obey that -- observed answering "who was
    Mahatma Gandhi" in full with zero retrieved sources. Instruction-following
    is not a safety mechanism at this size.

    So when there is no evidence, the model is not given the opportunity: this
    prompt contains no passages and permits only three replies. Removing the
    option is more reliable than forbidding it.
    """
    language = LANGUAGE_NAMES.get(request["language"], "English")
    return f"""You are StoryTutor, a tutor for NCERT class 6-8 Science and Social Science.

Reply in {language}.

You have NO textbook passages for this message. You searched and found nothing
relevant.

You may ONLY do one of these three things:

1. If the student greeted you or made small talk -- greet them back warmly and
   invite a question from their textbook.
2. If they asked what you are or what you can do -- say you answer questions
   from the NCERT class 6-8 Science and Social Science books in English, Hindi
   and Marathi, always showing which page the answer came from.
3. If they asked anything factual -- say you could not find it in those books.
   Suggest they rephrase, or pick the right class and subject.

You must NOT answer a factual question. Not from memory, not from general
knowledge, not "based on typical content", not even partially. If you know the
answer, you still must not give it -- an answer you cannot point to a page for
is exactly what this system exists to avoid.

Keep it to two or three sentences.

Return only valid JSON: {{"answer": "your reply here"}}"""


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

    # Routing is the model's job (see build_chat_prompt). A classifier only
    # runs when no model is reachable, so the fallback still greets sensibly.
    semantic = getattr(rag_engine, "intent_classifier", None)

    filters = {
        "class_level": request["class_level"],
        "subject": request["subject"],
        "language": request["language"],
    }
    results = rag_engine.retrieve(query=request["question"], filters=filters, top_k=top_k)

    strong = [doc for doc in results if _source_score(doc) >= MIN_EVIDENCE_SCORE]
    retrieval_method = results[0].get("retrieval_method") if results else None

    # With a model available, hand everything to it -- including greetings.
    if llm_client is not None:
        try:
            prompt = (build_chat_prompt(request, _format_sources(strong)) if strong
                      else build_no_source_prompt(request))
            answer = _generate(llm_client, request, prompt, history, prebuilt=True)
            if answer:
                checked = guardrails.check_output(answer, strong)
                return {
                    "answer": checked["answer"],
                    "sources": _citations(strong),
                    "grounded": bool(strong),
                    "model_provider": getattr(llm_client, "last_provider", "unknown"),
                    "retrieval_method": retrieval_method,
                    "intent": "model_routed",
                    "groundedness": checked["groundedness"],
                    "low_groundedness": checked["low_groundedness"],
                    "citations_used": checked["citations_used"],
                    "invented_citations": checked["invented_citations"],
                    **request,
                }
        except Exception as exc:  # noqa: BLE001
            print(f"chat: generation failed, falling back - {exc}")

    # --- no model: classify semantically so a greeting still gets a greeting ---
    if semantic is not None:
        intent, confidence = semantic.classify(request["question"])
    else:
        intent, confidence = classify_lexical(request["question"]), 0.0

    if intent != QUESTION:
        return {
            "answer": conversational_reply(intent, request["language"]),
            "sources": [], "grounded": True, "model_provider": "conversational",
            "retrieval_method": None, "intent": intent,
            "intent_confidence": round(confidence, 3),
            **request,
        }

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
              history: List[Dict[str, str]] = None, prebuilt: bool = False) -> Optional[str]:
    """Call the provider chain. `context` is a full prompt when prebuilt."""
    from story_mvp.model_clients import parse_json_response

    system = context if prebuilt else build_chat_prompt(request, context)
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


# --------------------------------------------------------------- streaming

def stream_answer(payload: Dict[str, Any], rag_engine, llm_client=None, top_k: int = 5):
    """Yield the answer in pieces as the model produces it.

    Total generation time is unchanged; what changes is that the student sees
    words in under a second instead of watching a spinner for eight. Perceived
    latency is the product here.

    Yields dicts: {"type": "meta"|"token"|"done"|"error", ...}
    """
    import time

    t0 = time.time()
    request = normalize_chat_request(payload)
    history = _recent_history(payload)

    allowed, reason, guard_meta = guardrails.check_input(request["question"])
    if not allowed:
        yield {"type": "meta", "sources": [], "intent": "blocked"}
        yield {"type": "token", "text": guardrails.refusal_message(reason, request["language"])}
        yield {"type": "done", "model_provider": "guardrail", "blocked_reason": reason}
        return
    if guard_meta["pii_types"]:
        request["question"] = guardrails.redact_pii(request["question"])

    filters = {
        "class_level": request["class_level"],
        "subject": request["subject"],
        "language": request["language"],
    }
    results = rag_engine.retrieve(query=request["question"], filters=filters, top_k=top_k)
    strong = [d for d in results if _source_score(d) >= MIN_EVIDENCE_SCORE]
    retrieval_ms = int((time.time() - t0) * 1000)

    # Sources go out first so citations can render while tokens still arrive.
    yield {
        "type": "meta",
        "sources": _citations(strong),
        "retrieval_method": results[0].get("retrieval_method") if results else None,
        "retrieval_ms": retrieval_ms,
    }

    if llm_client is None:
        text = (_passages_fallback(request, strong) if strong
                else NO_EVIDENCE.get(request["language"], NO_EVIDENCE["english"]))
        yield {"type": "token", "text": text}
        yield {"type": "done", "model_provider": "passages_only" if strong else "none",
               "retrieval_ms": retrieval_ms, "grounded": bool(strong)}
        return

    system = (build_chat_prompt(request, _format_sources(strong)) if strong
              else build_no_source_prompt(request))
    pieces: List[str] = []
    first_token_ms = None
    provider_name = "unknown"

    client = getattr(llm_client, "client", llm_client)
    for provider in getattr(client, "clients", [client]):
        pieces.clear()
        try:
            for piece in _stream_provider(provider, system, request["question"], history):
                if first_token_ms is None:
                    first_token_ms = int((time.time() - t0) * 1000)
                pieces.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:  # noqa: BLE001
            print(f"stream: {getattr(provider, 'provider_name', '?')} failed - {exc}")
            continue
        if pieces:
            provider_name = getattr(provider, "provider_name", "unknown")
            llm_client.last_provider = provider_name
            break

    if not pieces:
        text = (_passages_fallback(request, strong) if strong
                else NO_EVIDENCE.get(request["language"], NO_EVIDENCE["english"]))
        yield {"type": "token", "text": text}
        yield {"type": "done", "model_provider": "passages_only" if strong else "none",
               "retrieval_ms": retrieval_ms, "grounded": bool(strong)}
        return

    raw = "".join(pieces)
    try:
        answer = parse_json_response_safe(raw)
    except Exception:
        answer = raw
    checked = guardrails.check_output(answer, strong)

    yield {
        "type": "done",
        "model_provider": provider_name,
        "grounded": bool(strong),
        "answer": checked["answer"],
        "groundedness": checked["groundedness"],
        "citations_used": checked["citations_used"],
        "invented_citations": checked["invented_citations"],
        "retrieval_ms": retrieval_ms,
        "first_token_ms": first_token_ms,
        "total_ms": int((time.time() - t0) * 1000),
    }


def parse_json_response_safe(raw: str) -> str:
    """Pull the answer out of a JSON reply, or take the prose if it isn't JSON."""
    from story_mvp.model_clients import parse_json_response

    try:
        return parse_json_response(raw).get("answer", "").strip() or raw.strip()
    except Exception:
        return raw.strip()


def _stream_provider(provider, system: str, user: str, history=None):
    """Token stream from one provider. Ollama and OpenAI-compatible differ only
    in envelope shape."""
    import json
    import urllib.request

    name = getattr(provider, "provider_name", "")
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user})

    if name == "ollama":
        body = {"model": provider.model, "stream": True, "format": "json",
                "messages": messages, "options": {"temperature": 0.4}}
        req = urllib.request.Request(
            f"{provider.host}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=getattr(provider, "timeout", 180)) as r:
            for line in r:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return
        return

    base = getattr(provider, "base_url", None) or getattr(provider, "endpoint", "")
    url = f"{base}/chat/completions" if base and not base.endswith("/chat/completions") else base
    body = {"model": provider.model, "messages": messages, "temperature": 0.4,
            "max_tokens": 1200, "stream": True}
    headers = {"Content-Type": "application/json"}
    key = getattr(provider, "api_key", None)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=getattr(provider, "timeout", 180)) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except Exception:
                continue
            piece = delta.get("content", "")
            if piece:
                yield piece
