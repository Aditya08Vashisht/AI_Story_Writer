"""Local story generation engine for the Kuku FMs MVP demo.

This module intentionally avoids network calls and heavyweight model loading.
It gives the demo a dependable creative surface now, while keeping the API
shape close to a future fine-tuned LLM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Dict, List


GENRES: Dict[str, Dict[str, object]] = {
    "thriller": {
        "sensory": ["a locked phone buzzing under the floorboards", "rain ticking against a broken window", "a corridor smelling of wet concrete"],
        "stakes": "a secret that could ruin an entire family",
        "setting": "a half-lit apartment above a silent marketplace",
        "turn": "the safest witness is the one who has been lying from the start",
    },
    "romance": {
        "sensory": ["a paper cup warming cold fingers", "the old song playing from a tea stall", "a message left unsent at midnight"],
        "stakes": "a love story that arrives at the worst possible time",
        "setting": "a city terrace after the first rain",
        "turn": "the person walking away has already written the ending",
    },
    "horror": {
        "sensory": ["a prayer bell ringing by itself", "footsteps stopping outside a locked room", "a childhood rhyme whispered through static"],
        "stakes": "a fear that knows every name in the house",
        "setting": "an ancestral home where every mirror is covered",
        "turn": "the ghost is not asking for revenge, but for help",
    },
    "fantasy": {
        "sensory": ["ashes glowing like fireflies", "a map that redraws itself in moonlight", "the taste of iron before magic wakes"],
        "stakes": "a kingdom balanced on one forbidden choice",
        "setting": "a border town built around a sleeping dragon shrine",
        "turn": "the chosen one is only a decoy for the real heir",
    },
    "family drama": {
        "sensory": ["pressure cooker whistles from the kitchen", "old photo albums smelling of dust", "a festival light flickering above an empty chair"],
        "stakes": "a truth that can either heal the family or split it forever",
        "setting": "a crowded home on the morning of a wedding",
        "turn": "the person everyone blames is the only one protecting them",
    },
    "comedy": {
        "sensory": ["a microphone squealing at exactly the wrong moment", "chai spilling on a printed master plan", "a fake moustache sliding loose"],
        "stakes": "a tiny lie growing into a full public disaster",
        "setting": "a community hall five minutes before the chief guest arrives",
        "turn": "the worst plan in the room accidentally solves everything",
    },
}

TONES: Dict[str, Dict[str, str]] = {
    "cinematic": {
        "pace": "cutting between intimate detail and big-screen tension",
        "texture": "visual, atmospheric, and ready for voice performance",
    },
    "suspenseful": {
        "pace": "slow-burning, with each line tightening the question",
        "texture": "uneasy, restrained, and built around delayed reveals",
    },
    "emotional": {
        "pace": "quietly intense, giving each decision emotional weight",
        "texture": "warm, intimate, and character-first",
    },
    "witty": {
        "pace": "quick, playful, and powered by reversals",
        "texture": "sharp, conversational, and easy to perform",
    },
    "mythic": {
        "pace": "grand but clear, with a sense of fate pressing in",
        "texture": "elevated, image-rich, and dramatic",
    },
}

LANGUAGE_NOTES = {
    "english": "English",
    "hindi": "Hindi-ready English draft",
    "hinglish": "Hinglish-flavored draft",
    "marathi": "Marathi-ready draft",
}

MODE_TITLES = {
    "hook": "Hook Generator",
    "expand": "Scene Expander",
    "continue": "Story Continuation",
}

SAFE_MODE_FALLBACK = "continue"
SAFE_GENRE_FALLBACK = "thriller"
SAFE_TONE_FALLBACK = "cinematic"


@dataclass(frozen=True)
class StoryRequest:
    mode: str
    idea: str
    genre: str = SAFE_GENRE_FALLBACK
    tone: str = SAFE_TONE_FALLBACK
    language: str = "english"
    characters: str = ""
    length: str = "medium"
    class_level: str = ""
    subject: str = ""
    chapter: str = ""
    topic: str = ""
    output_type: str = "study_story"
    difficulty: str = "medium"


def generate_story_piece(payload: Dict[str, str]) -> Dict[str, object]:
    """Generate a demo story artifact from a frontend/API payload."""
    request = _normalize_request(payload)
    genre_data = GENRES[request.genre]
    tone_data = TONES[request.tone]
    seed = _seed(request)
    cast = _extract_characters(request.characters, request.idea)

    if request.mode == "hook":
        output = _generate_hook(request, genre_data, tone_data, cast, seed)
    elif request.mode == "expand":
        output = _generate_scene(request, genre_data, tone_data, cast, seed)
    else:
        output = _generate_continuation(request, genre_data, tone_data, cast, seed)

    title = _title_from_idea(request.idea, request.genre)
    result = {
        "title": title,
        "mode": request.mode,
        "mode_label": MODE_TITLES[request.mode],
        "genre": request.genre,
        "tone": request.tone,
        "language": request.language,
        "output": output,
        "pitch": _pitch_line(request, genre_data, cast),
        "style_notes": _style_notes(request, tone_data),
        "story_bible": _story_bible(request, genre_data, cast),
        "safety": _safety_check(request.idea),
    }
    result.update(_educational_artifacts(request, output, [], provider="deterministic"))
    result["retrieved_sources"] = []
    return result

def generate_story_piece_ai(payload: Dict[str, str], rag_engine, llm_client) -> Dict[str, object]:
    """Generate a story piece using RAG and LLM."""
    request = _normalize_request(payload)
    
    filters = {
        "class_level": request.class_level,
        "subject": request.subject,
        "language": request.language if request.language in {"english", "hindi", "marathi"} else "",
        "chapter": request.chapter,
        "topic": request.topic,
    }
    curriculum_query = " ".join(
        part for part in (request.idea, request.chapter, request.topic) if part
    )
    results = rag_engine.retrieve(query=curriculum_query, filters=filters)
    context_text = rag_engine.get_context_text(results)
    
    try:
        llm_response = llm_client.generate(request, context_text)
    except Exception as e:
        print(f"Fallback triggered: LLM failed - {e}")
        fallback = generate_story_piece(payload)
        fallback["retrieved_sources"] = _source_documents(rag_engine, results)
        return fallback
        
    title = _title_from_idea(request.idea, request.genre)
    safety_info = _safety_check(llm_response.get("output", ""))
    result = {
        "title": title,
        "mode": request.mode,
        "mode_label": MODE_TITLES[request.mode],
        "genre": request.genre,
        "tone": request.tone,
        "language": request.language,
        "output": llm_response.get("output", "No output generated."),
        "pitch": llm_response.get("pitch", ""),
        "style_notes": llm_response.get("style_notes", []),
        "story_bible": llm_response.get("story_bible", {}),
        "safety": safety_info,
    }
    result.update(_educational_artifacts(request, result["output"], results, llm_response, getattr(llm_client, "last_provider", "unknown")))
    result["retrieved_sources"] = _source_documents(rag_engine, results)
    return result


def _normalize_request(payload: Dict[str, str]) -> StoryRequest:
    mode = _clean_choice(payload.get("mode", ""), MODE_TITLES.keys(), SAFE_MODE_FALLBACK)
    genre = _clean_choice(payload.get("genre", ""), GENRES.keys(), SAFE_GENRE_FALLBACK)
    tone = _clean_choice(payload.get("tone", ""), TONES.keys(), SAFE_TONE_FALLBACK)
    language = _clean_choice(payload.get("language", ""), LANGUAGE_NOTES.keys(), "english")
    length = _clean_choice(payload.get("length", ""), {"short", "medium", "long"}, "medium")
    idea = _clean_text(payload.get("idea", ""))
    characters = _clean_text(payload.get("characters", ""))
    class_level = _clean_choice(payload.get("class_level", ""), {"6", "7", "8"}, "")
    subject = _clean_choice(payload.get("subject", ""), {"science", "social_science", "social science", "sst"}, "")
    if subject in {"social science", "sst"}:
        subject = "social_science"
    chapter = _clean_text(payload.get("chapter", ""))[:120]
    topic = _clean_text(payload.get("topic", ""))[:160]
    output_type = _clean_choice(
        payload.get("output_type", ""),
        {"study_story", "video_demo_plan", "explanation", "quiz", "story_video_plan"},
        "study_story",
    )
    difficulty = _clean_choice(payload.get("difficulty", ""), {"easy", "medium", "hard"}, "medium")

    if not idea:
        idea = "Help a student understand a curriculum concept through a clear story and explanation."

    return StoryRequest(
        mode=mode,
        idea=idea,
        genre=genre,
        tone=tone,
        language=language,
        characters=characters,
        length=length,
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        topic=topic,
        output_type=output_type,
        difficulty=difficulty,
    )


def _clean_choice(value: str, allowed, fallback: str) -> str:
    normalized = _clean_text(value).lower()
    return normalized if normalized in allowed else fallback


def _clean_text(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:1200]


def _seed(request: StoryRequest) -> int:
    digest = sha256(f"{request.mode}|{request.genre}|{request.tone}|{request.idea}|{request.characters}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pick(items: List[str], seed: int, offset: int = 0) -> str:
    return items[(seed + offset) % len(items)]


def _extract_characters(characters: str, idea: str) -> List[str]:
    if characters:
        names = [part.strip() for part in re.split(r"[,;/]", characters) if part.strip()]
    else:
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", idea)

    filtered = []
    for name in names:
        clean = re.sub(r"[^A-Za-z ]", "", name).strip()
        if clean and clean.lower() not in {"the", "and", "but", "one"} and clean not in filtered:
            filtered.append(clean)

    return filtered[:4] or ["Aarav", "Meera"]


def _generate_hook(request: StoryRequest, genre_data: Dict[str, object], tone_data: Dict[str, str], cast: List[str], seed: int) -> str:
    sensory = _pick(genre_data["sensory"], seed)
    lead = cast[0]
    foil = cast[1] if len(cast) > 1 else "the only person who believes them"
    base = [
        f"When {lead} discovers {sensory}, {foil} insists it is nothing, until the same warning appears inside tomorrow's audio script.",
        f"{lead} has one night to turn a half-forgotten idea into a hit story, but every draft reveals {genre_data['stakes']}.",
        f"The first episode begins with a simple promise: nobody will die tonight. By the ad break, {lead} knows the promise was a trap.",
    ]
    hook = _pick(base, seed, 3)
    return _language_wrap(request.language, hook)


def _generate_scene(request: StoryRequest, genre_data: Dict[str, object], tone_data: Dict[str, str], cast: List[str], seed: int) -> str:
    lead = cast[0]
    second = cast[1] if len(cast) > 1 else "Meera"
    sensory = _pick(genre_data["sensory"], seed)
    setting = genre_data["setting"]
    turn = genre_data["turn"]
    length_hint = {
        "short": 2,
        "medium": 3,
        "long": 4,
    }[request.length]

    paragraphs = [
        f"{lead} reached {setting} with the feeling that the place had been waiting. {sensory.capitalize()} cut through the ordinary noise around them, turning every small movement into a warning. The idea was simple on paper: {request.idea}. In the room, it felt much larger, as if the story had found a body and started breathing.",
        f"{second} tried to laugh it off, but the laugh came out too thin. \"Say exactly what you saw,\" {second} said. {lead} looked at the doorway, then at the shadow underneath it. The answer should have been easy. Instead, {lead} noticed the one detail that did not belong: the scene was already arranged like the final page of a script.",
        f"The moment broke when a new sound entered the room. Not loud, not dramatic, just precise. It forced both of them to understand the same thing at once: {turn}. {lead} did not run. A better instinct took over. If this was a story, then somebody had written a clue into the fear.",
        f"By the end of the scene, {lead} makes a choice that changes the episode engine: protect the secret for one more night, or expose it and lose the only person still willing to listen.",
    ]
    return _language_wrap(request.language, "\n\n".join(paragraphs[:length_hint]))


def _generate_continuation(request: StoryRequest, genre_data: Dict[str, object], tone_data: Dict[str, str], cast: List[str], seed: int) -> str:
    lead = cast[0]
    second = cast[1] if len(cast) > 1 else "Kabir"
    sensory = _pick(genre_data["sensory"], seed, 1)
    stakes = genre_data["stakes"]
    beats = [
        f"{lead} does not answer immediately. The silence gives the room enough time to change. Somewhere nearby, {sensory}, and {second} realizes the old version of the plan is useless now.",
        f"Instead of explaining, {lead} opens the last message again. The words are the same, but the meaning is not. It is no longer a warning. It is an invitation.",
        f"They decide to follow the clue before sunrise. That choice keeps the story moving and raises the central question: who benefits if {stakes} stays buried?",
        f"The episode can end on a clean cliffhanger: {second} finds proof that {lead} was present at the beginning of the mystery, even though {lead} has no memory of being there.",
    ]
    count = {"short": 2, "medium": 3, "long": 4}[request.length]
    return _language_wrap(request.language, "\n\n".join(beats[:count]))


def _language_wrap(language: str, text: str) -> str:
    if language == "hinglish":
        return f"{text}\n\nPerformance flavor: keep the narration English-led, but let emotional dialogue land in natural Hinglish where it feels intimate."
    if language == "hindi":
        return f"{text}\n\nHindi adaptation note: preserve the beats, translate dialogue naturally, and keep narration suitable for Hindi audio drama."
    if language == "marathi":
        return f"{text}\n\nMarathi adaptation note: preserve the concept accurately, use simple Marathi for narration, and keep important textbook terms clear."
    return text


def _title_from_idea(idea: str, genre: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", idea)
    meaningful = [w for w in words if len(w) > 3][:4]
    if meaningful:
        return " ".join(word.capitalize() for word in meaningful)
    return f"Untitled {genre.title()}"


def _pitch_line(request: StoryRequest, genre_data: Dict[str, object], cast: List[str]) -> str:
    return f"A {request.tone} {request.genre} built around {cast[0]}'s choice, {genre_data['stakes']}, and an audio-first cliffhanger."


def _style_notes(request: StoryRequest, tone_data: Dict[str, str]) -> List[str]:
    return [
        f"Voice: {tone_data['texture']}.",
        f"Pacing: {tone_data['pace']}.",
        f"Format: optimized for serialized listening, with clear scene turns and cliffhanger potential.",
        f"Language mode: {LANGUAGE_NOTES[request.language]}.",
    ]


def _story_bible(request: StoryRequest, genre_data: Dict[str, object], cast: List[str]) -> Dict[str, object]:
    return {
        "central_conflict": genre_data["stakes"],
        "primary_characters": cast,
        "recurring_image": _pick(genre_data["sensory"], _seed(request), 2),
        "next_episode_question": f"What does {cast[0]} lose if the truth comes out now?",
    }


def _safety_check(text: str) -> Dict[str, object]:
    lowered = text.lower()
    flagged_terms = [term for term in ["self-harm", "sexual minor", "hate crime"] if term in lowered]
    return {
        "status": "review" if flagged_terms else "clear",
        "flags": flagged_terms,
        "note": "Demo filter only. Production should use a dedicated moderation model and policy layer.",
    }


def _educational_artifacts(
    request: StoryRequest,
    output: str,
    sources: List[Dict],
    llm_response: Dict[str, object] = None,
    provider: str = "deterministic",
) -> Dict[str, object]:
    llm_response = llm_response or {}
    learning_objective = llm_response.get("learning_objective") or _learning_objective(request)
    explanation = llm_response.get("explanation") or _fallback_explanation(request, sources)
    story = llm_response.get("story") or output
    storyboard = llm_response.get("storyboard") or _storyboard(request, explanation, story)
    diagram_plan = llm_response.get("diagram_plan") or _diagram_plan(request)
    narration_script = llm_response.get("narration_script") or _narration_script(request, storyboard)
    video_demo_plan = llm_response.get("video_demo_plan") or _video_demo_plan(storyboard, diagram_plan)
    return {
        "learning_objective": learning_objective,
        "explanation": explanation,
        "story": story,
        "quiz": llm_response.get("quiz") or _quiz(request),
        "storyboard": storyboard,
        "diagram_plan": diagram_plan,
        "narration_script": narration_script,
        "video_demo_plan": video_demo_plan,
        "model_provider": provider,
        "free_first": True,
    }


def _learning_objective(request: StoryRequest) -> str:
    subject = request.subject.replace("_", " ") if request.subject else "the subject"
    level = f"Class {request.class_level} " if request.class_level else ""
    return f"Help a {level}student understand {request.idea} in {subject} using story, explanation, and a quick check."


def _fallback_explanation(request: StoryRequest, sources: List[Dict]) -> str:
    source_hint = sources[0].get("text", "")[:280] if sources else request.idea
    return _language_wrap(
        request.language,
        f"Concept explanation: {request.idea}. Start with the observable situation, name the key idea, connect it to the curriculum source, and ask the learner to explain the idea back in their own words. Source clue: {source_hint}",
    )


def _storyboard(request: StoryRequest, explanation: str, story: str) -> List[Dict[str, object]]:
    return [
        {
            "scene_number": 1,
            "visual": f"Title slide for {request.idea}",
            "narration": f"Today we will learn: {request.idea}",
            "learning_point": "Introduce the learning goal.",
        },
        {
            "scene_number": 2,
            "visual": "Simple story scene with learner characters and one clear phenomenon.",
            "narration": story[:280],
            "learning_point": "Make the concept memorable through story.",
        },
        {
            "scene_number": 3,
            "visual": "Diagram or flowchart that connects the main ideas.",
            "narration": explanation[:280],
            "learning_point": "Explain the concept clearly.",
        },
        {
            "scene_number": 4,
            "visual": "Quiz card with one question and four choices.",
            "narration": "Pause and answer the check question before moving ahead.",
            "learning_point": "Check understanding.",
        },
    ]


def _diagram_plan(request: StoryRequest) -> Dict[str, object]:
    topic = request.topic or request.chapter or request.idea
    return {
        "type": "flowchart",
        "title": f"{topic} concept map",
        "nodes": ["Learning goal", "Story example", "Concept explanation", "Misconception check", "Quiz"],
        "edges": [
            ["Learning goal", "Story example"],
            ["Story example", "Concept explanation"],
            ["Concept explanation", "Misconception check"],
            ["Misconception check", "Quiz"],
        ],
    }


def _narration_script(request: StoryRequest, storyboard: List[Dict[str, object]]) -> str:
    lines = [f"Scene {scene['scene_number']}: {scene['narration']}" for scene in storyboard]
    return _language_wrap(request.language, "\n".join(lines))


def _video_demo_plan(storyboard: List[Dict[str, object]], diagram_plan: Dict[str, object]) -> Dict[str, object]:
    return {
        "scenes": storyboard,
        "assets_needed": [
            "Generated title slide",
            "Simple illustrated story frame",
            f"{diagram_plan.get('type', 'diagram')} diagram rendered locally",
            "Quiz card",
            "Text narration script",
        ],
        "renderable_without_paid_api": True,
        "recommended_free_tools": ["ffmpeg", "moviepy", "HTML/SVG diagrams", "browser screenshots"],
    }


def _quiz(request: StoryRequest) -> List[Dict[str, object]]:
    return [
        {
            "question": f"Which statement best explains the learning goal: {request.idea}?",
            "options": [
                "It connects evidence or examples to the concept.",
                "It only repeats the story without explaining.",
                "It ignores the textbook source.",
                "It changes the topic completely.",
            ],
            "answer": "It connects evidence or examples to the concept.",
            "explanation": "A good study story must still explain the curriculum concept accurately.",
        }
    ]


def _source_documents(rag_engine, results: List[Dict]) -> List[Dict[str, object]]:
    if hasattr(rag_engine, "source_documents"):
        return rag_engine.source_documents(results)
    return [{"id": doc.get("id"), "source_file": doc.get("source_file")} for doc in results]
