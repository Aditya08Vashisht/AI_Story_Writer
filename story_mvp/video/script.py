"""Turn a grounded RAG answer into a validated 4-scene video script.

The hard part is not asking the model for a script -- it is refusing the ones
that would produce a bad video. Three things are enforced in code rather than
requested in the prompt:

  word budgets   30 seconds IS 70-80 words, because narration runs ~2.5 w/s in
                 English and ~2.2 in Hindi/Marathi. A model asked politely for
                 "short" narration will happily write 200 words.
  citations      a [S2] marker when only one source was retrieved fails the
                 build. It must never reach a rendered frame, where it looks
                 authoritative.
  grounding      no retrieved sources means no video. A polished, confident,
                 wrong video is worse than no video, because video is more
                 persuasive than text.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Narration speed, words per second. Devanagari is slower to speak.
WORDS_PER_SECOND = {"english": 2.5, "hindi": 2.2, "marathi": 2.2}

# scene key -> (seconds, max words)
SCENE_BUDGET = {
    "title": (4.0, 10),
    "idea": (9.0, 22),
    "diagram": (11.0, 26),
    "check": (6.0, 14),
}
TOTAL_SECONDS = sum(v[0] for v in SCENE_BUDGET.values())

SCRIPT_INSTRUCTION = (
    "Write the video script now. Reply with the JSON object only, using exactly "
    "the keys shown: title, scenes (title, idea, diagram, check -- each with "
    "heading, body, narration), diagram_nodes, diagram_edges. Do not answer the "
    "question directly and do not add an 'answer' key."
)

# A one-word overrun should be sent back for a retry, not treated as fatal.
# Rejecting 15 words against a limit of 14 cost a whole attempt on Sol. Total
# length stays bounded: 72 budgeted words become at most ~86.
WORD_SLACK = 1.2

_CITATION = re.compile(r"\[S(\d+)\]")
_WORD = re.compile(r"[\wऀ-ॣ०-ॿ]+")


def count_words(text: str) -> int:
    return len(_WORD.findall(text or ""))


@dataclass
class Scene:
    key: str
    heading: str
    body: str
    narration: str
    seconds: float


@dataclass
class VideoScript:
    question: str
    language: str
    class_level: str
    subject: str
    title: str
    scenes: List[Scene]
    diagram_nodes: List[str]
    diagram_edges: List[List[str]]
    source_line: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model_provider: str = "unknown"

    @property
    def total_words(self) -> int:
        return sum(count_words(s.narration) for s in self.scenes)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_words"] = self.total_words
        d["estimated_seconds"] = round(sum(s.seconds for s in self.scenes), 1)
        return d


class ScriptError(RuntimeError):
    pass


def build_script_prompt(question: str, answer: str, sources: List[Dict], request: Dict[str, str]) -> str:
    language = {"english": "English", "hindi": "Hindi", "marathi": "Marathi"}.get(
        request.get("language", "english"), "English")
    n = len(sources)
    budgets = "\n".join(
        f"  {k}: at most {w} words of narration" for k, (_, w) in SCENE_BUDGET.items()
    )
    return f"""You are writing a 30-second explainer video for an Indian school student.

The text VALUES you write must be in {language}. The JSON KEYS must stay
exactly as shown below, in English -- "title", "scenes", "heading", "body",
"narration", "diagram_nodes", "diagram_edges". Do not translate the keys.

Here is a grounded answer, already checked against NCERT textbook pages:
---
{answer}
---

There are {n} source(s), numbered [S1] to [S{n}]. Never use a higher number.

Write a four-scene script. The word budgets are hard limits, not suggestions --
narration is spoken aloud at about 2.3 words per second, so exceeding them
makes the video run long and the scenes desync:

{budgets}

Also give a simple concept diagram: 2 to 5 short node labels and the arrows
between them, showing how the idea flows. Node labels must be 1-4 words.

Return only valid JSON in exactly this shape:
{{
  "title": "short video title, at most 6 words",
  "scenes": {{
    "title":   {{"heading": "...", "body": "...", "narration": "..."}},
    "idea":    {{"heading": "...", "body": "...", "narration": "..."}},
    "diagram": {{"heading": "...", "body": "...", "narration": "..."}},
    "check":   {{"heading": "...", "body": "a question for the student", "narration": "..."}}
  }},
  "diagram_nodes": ["...", "...", "..."],
  "diagram_edges": [["node a", "node b"], ["node b", "node c"]]
}}

"body" is what appears on screen -- keep it under 14 words, it must fit a card.
"narration" is what is spoken. They should agree but need not be identical."""


def validate(script: VideoScript, n_sources: int) -> List[str]:
    """Return a list of problems. Empty means the script is usable."""
    problems: List[str] = []

    for scene in script.scenes:
        budget = SCENE_BUDGET.get(scene.key, (0, 25))[1]
        limit = int(budget * WORD_SLACK + 0.999)
        words = count_words(scene.narration)
        if words > limit:
            problems.append(f"scene '{scene.key}': {words} words of narration, limit {limit}")
        if not scene.narration.strip():
            problems.append(f"scene '{scene.key}': empty narration")
        if count_words(scene.body) > 20:
            problems.append(f"scene '{scene.key}': on-screen body too long for a card")

        for marker in _CITATION.findall(scene.narration + " " + scene.body):
            if not 1 <= int(marker) <= n_sources:
                problems.append(f"scene '{scene.key}': cites [S{marker}] but only {n_sources} source(s) exist")

    # Two nodes is a legitimate diagram -- "hot water -> spoon" is the whole
    # idea for the heat-transfer concept. Requiring three rejected it three
    # times on Sol and refused a video that was otherwise correct.
    if not 2 <= len(script.diagram_nodes) <= 6:
        problems.append(f"diagram needs 2-6 nodes, got {len(script.diagram_nodes)}")

    known = {n.strip().lower() for n in script.diagram_nodes}
    for edge in script.diagram_edges:
        if len(edge) != 2:
            problems.append(f"malformed edge: {edge}")
            continue
        for end in edge:
            if str(end).strip().lower() not in known:
                problems.append(f"edge references unknown node: {end!r}")

    return problems


def _source_line(sources: List[Dict], language: str) -> str:
    if not sources:
        return ""
    s = sources[0]
    bits = []
    if s.get("class_level"):
        bits.append(f"Class {s['class_level']}")
    if s.get("subject"):
        bits.append(s["subject"].replace("_", " ").title())
    if s.get("chapter"):
        bits.append(f"Ch. {s['chapter']}")
    if s.get("page"):
        bits.append(f"p. {s['page']}")
    return "NCERT " + ", ".join(bits) if bits else "NCERT"


def script_from_answer(
    rag_result: Dict[str, Any],
    llm_client,
    request: Dict[str, str],
    max_attempts: int = 3,
    debug_dir: Optional[Any] = None,
) -> VideoScript:
    """Build a validated script, retrying with the problems fed back.

    Every rejected reply is written to `debug_dir` when given. The Hindi and
    Marathi failures were diagnosed by inference because the raw reply was
    never visible; it should not take guessing a second time.
    """
    sources = rag_result.get("sources") or []
    if not sources or not rag_result.get("grounded"):
        raise ScriptError(
            "No grounded sources for this question, so no video. A confident "
            "video built on nothing is worse than no video."
        )

    from story_mvp.rag_chat import _call_provider
    from story_mvp.model_clients import parse_json_response

    base_prompt = build_script_prompt(
        rag_result.get("question", ""), rag_result.get("answer", ""), sources, request
    )
    client = getattr(llm_client, "client", llm_client)
    providers = list(getattr(client, "clients", [client]))

    last_problems: List[str] = []
    for attempt in range(max_attempts):
        prompt = base_prompt
        if last_problems:
            # Feed the failures back rather than retrying blind.
            issues = "\n".join("- " + p for p in last_problems)
            prompt += (
                "\n\nYour previous attempt had these problems. Fix all of them:\n" + issues
            )

        raw = None
        provider_name = "unknown"
        for provider in providers:
            try:
                # The user turn is a fixed instruction, NOT the student's
                # question. Sending the question made the model answer it --
                # replying {"answer": ...} with no scenes at all -- which is
                # why every Hindi and Marathi attempt came back identically
                # empty. The question is already inside the system prompt.
                raw = _call_provider(provider, prompt, SCRIPT_INSTRUCTION, None)
                provider_name = getattr(provider, "provider_name", "unknown")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"video script: {getattr(provider,'provider_name','?')} failed - {exc}")
        if raw is None:
            raise ScriptError("no model could produce a script")

        try:
            data = parse_json_response(raw)
        except Exception:
            try:
                data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            except Exception as exc:  # noqa: BLE001
                last_problems = [f"reply was not valid JSON: {exc}"]
                _dump_reply(debug_dir, request, attempt + 1, raw, last_problems)
                continue

        script = _assemble(data, rag_result, request, sources, provider_name)
        problems = validate(script, len(sources))
        if not problems:
            return script
        last_problems = problems
        print(f"video script attempt {attempt+1} rejected: {problems}")
        _dump_reply(debug_dir, request, attempt + 1, raw, problems)

    raise ScriptError(f"could not produce a valid script in {max_attempts} attempts: {last_problems}")


def _dump_reply(debug_dir, request: Dict[str, str], attempt: int, raw: str, problems: List[str]) -> None:
    if not debug_dir:
        return
    from pathlib import Path

    try:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in request.get("question", "q"))[:40]
        (d / f"{request.get('language','x')}_{slug}_attempt{attempt}.txt").write_text(
            "PROBLEMS:\n" + "\n".join(problems) + "\n\nRAW REPLY:\n" + (raw or ""),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"could not save debug reply - {exc}")


_SCENE_FIELDS = ("heading", "body", "narration")


def recover_structure(data: Any) -> Dict[str, Any]:
    """Map a reply onto the expected shape when the model got the keys wrong.

    Observed with qwen2.5:7b on Hindi and Marathi: the reply parsed as valid
    JSON yet mapped to nothing -- every scene empty, zero diagram nodes, the
    same on all three attempts even with the problems fed back. A model
    writing a poor script produces a different mistake each time; producing
    nothing identically means the content was there under keys the parser
    did not recognise, most likely translated along with the values.

    Recovery is positional and only fills keys that are MISSING. It never
    invents content -- every string still comes from the model, and the
    result still has to pass validate() before anything is rendered.
    """
    if isinstance(data, list):
        data = {"scenes": data} if data and all(isinstance(x, dict) for x in data) else {}
    if not isinstance(data, dict):
        return {}

    out = dict(data)
    order = list(SCENE_BUDGET)

    scenes = data.get("scenes")
    if scenes is None:
        for v in data.values():
            if isinstance(v, dict) and len(v) >= len(order) and all(isinstance(x, dict) for x in v.values()):
                scenes = v
                break
            if isinstance(v, list) and len(v) >= len(order) and all(isinstance(x, dict) for x in v):
                scenes = v
                break

    if isinstance(scenes, list):
        scenes = {k: scenes[i] for i, k in enumerate(order) if i < len(scenes)}
    elif isinstance(scenes, dict) and not all(k in scenes for k in order):
        vals = list(scenes.values())
        scenes = {k: vals[i] for i, k in enumerate(order) if i < len(vals)}
    elif not isinstance(scenes, dict):
        scenes = {}

    fixed = {}
    for k, s in scenes.items():
        if not isinstance(s, dict):
            continue
        if not any(f in s for f in _SCENE_FIELDS):
            vals = [str(v) for v in s.values()]
            s = {f: vals[i] for i, f in enumerate(_SCENE_FIELDS) if i < len(vals)}
        fixed[k] = s
    out["scenes"] = fixed

    if not out.get("diagram_nodes"):
        for v in data.values():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                out["diagram_nodes"] = v
                break
    if not out.get("diagram_edges"):
        for v in data.values():
            if isinstance(v, list) and v and all(isinstance(x, list) and len(x) == 2 for x in v):
                out["diagram_edges"] = v
                break
    if not out.get("title"):
        for v in data.values():
            if isinstance(v, str) and v.strip():
                out["title"] = v
                break
    return out


def _assemble(data: Dict, rag_result: Dict, request: Dict[str, str],
              sources: List[Dict], provider: str) -> VideoScript:
    data = recover_structure(data)
    raw_scenes = data.get("scenes") or {}
    scenes: List[Scene] = []
    for key, (seconds, _limit) in SCENE_BUDGET.items():
        s = raw_scenes.get(key) or {}
        scenes.append(Scene(
            key=key,
            heading=str(s.get("heading", "")).strip(),
            body=str(s.get("body", "")).strip(),
            narration=str(s.get("narration", "")).strip(),
            seconds=seconds,
        ))

    nodes = [str(n).strip() for n in (data.get("diagram_nodes") or []) if str(n).strip()]
    edges = [[str(a).strip(), str(b).strip()] for a, b in
             ((list(e) + ["", ""])[:2] for e in (data.get("diagram_edges") or [])
              if isinstance(e, (list, tuple))) if str(a).strip()]

    # A model often names nodes in its edges without listing them -- observed:
    # nodes ["Oceans cover 71%", ...] with edges ["Oceans", "Continents"]. The
    # endpoints are the model's own labels, so adding them to the node list is
    # faithful rather than invented. Capped at 6 so the card still fits.
    known = {n.lower() for n in nodes}
    for a, b in edges:
        for end in (a, b):
            if end and end.lower() not in known and len(nodes) < 6:
                nodes.append(end)
                known.add(end.lower())

    return VideoScript(
        question=rag_result.get("question", ""),
        language=request.get("language", "english"),
        class_level=request.get("class_level", ""),
        subject=request.get("subject", ""),
        title=str(data.get("title", "")).strip() or rag_result.get("question", "")[:60],
        scenes=scenes,
        diagram_nodes=nodes,
        diagram_edges=edges,
        source_line=_source_line(sources, request.get("language", "english")),
        sources=sources,
        model_provider=provider,
    )
