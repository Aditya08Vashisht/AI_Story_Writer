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

# The floor, not only the ceiling. With no minimum, a Hindi script of four
# words in total -- one per scene -- passed validation on Sol and made a
# 30-second video that explained nothing. About half of each budget.
SCENE_MIN_WORDS = {"title": 4, "idea": 10, "diagram": 12, "check": 6}

SCRIPT_INSTRUCTION = (
    "Write the video script now. Reply with the JSON object only, using exactly "
    "the keys shown in the instructions: title, scenes (title, idea, diagram, "
    "check), diagram_nodes, diagram_edges. Do not answer the question directly "
    "and do not add an 'answer' key."
)

# A one-word overrun should be sent back for a retry, not treated as fatal.
# Rejecting 15 words against a limit of 14 cost a whole attempt on Sol. Total
# length stays bounded: 72 budgeted words become at most ~86.
WORD_SLACK = 1.2

_CITATION = re.compile(r"\[S(\d+)\]")
_DEVA = re.compile(r"[ऀ-ॿ]")
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
    # English description of an illustration for this moment. Always English,
    # even for Hindi/Marathi videos, because that is what the image model reads.
    visual: str = ""


# Scenes that get a generated illustration. The diagram scene is drawn by
# Pillow from nodes and edges, because a picture must never carry a fact.
VISUAL_SCENES = ("title", "idea", "check")


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
    # How the script was checked against the grounded answer (see
    # grounding_problems). Lands in the sidecar, so every video says whether
    # its narration and diagram were verified, and by what.
    grounding: Dict[str, Any] = field(default_factory=dict)

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


def _scene_template(want_visuals: bool) -> str:
    rows = []
    for key in SCENE_BUDGET:
        body = "a question for the student" if key == "check" else "..."
        visual = ', "visual": "..."' if want_visuals and key in VISUAL_SCENES else ""
        rows.append(f'    "{key}": {{"heading": "...", "body": "{body}", "narration": "..."{visual}}}')
    return "{\n" + ",\n".join(rows) + "\n  }"


def _visual_rules(language: str) -> str:
    return f"""
The scenes "title", "idea" and "check" each also need a "visual": ONE sentence,
IN ENGLISH even though the video is in {language}, describing an illustration a
painter could draw for that moment -- concrete objects in a setting an Indian
student would recognise.

Never ask for words, letters, labels, signs, numbers, charts, maps or diagrams
in a visual. All text is added separately, and a picture must not carry a fact
that could be wrong -- the facts live in the narration and the diagram.
  Good: "a steel spoon resting in a cup of steaming chai on a kitchen counter"
  Good: "green leaves of a mango tree glowing in bright morning sunlight"
  Bad:  "a diagram labelled heat flow"      Bad: "a world map showing oceans"
"""


def build_script_prompt(question: str, answer: str, sources: List[Dict], request: Dict[str, str],
                        want_visuals: bool = False) -> str:
    language = {"english": "English", "hindi": "Hindi", "marathi": "Marathi"}.get(
        request.get("language", "english"), "English")
    n = len(sources)
    budgets = "\n".join(
        f"  {k}: {SCENE_MIN_WORDS[k]} to {w} words of narration"
        for k, (_, w) in SCENE_BUDGET.items()
    )
    visual_rules = _visual_rules(language) if want_visuals else ""
    keys = '"title", "scenes", "heading", "body", "narration", ' + ('"visual", ' if want_visuals else "") \
        + '"diagram_nodes", "diagram_edges"'
    scenes_json = _scene_template(want_visuals)
    # The question is the topic anchor. It was once left out of this prompt
    # entirely (the user turn is a fixed instruction, see SCRIPT_INSTRUCTION),
    # so the model saw only the answer and titled videos after whatever else
    # the answer happened to mention -- a title about one thing over scenes
    # about another.
    return f"""You are writing a 30-second explainer video for an Indian school student.

THE TOPIC OF THIS VIDEO -- the student's question:
    {question}
The title and every scene must be about exactly this question. Do not drift to
other topics that the answer or its sources also mention.

The text VALUES you write must be in {language}. The JSON KEYS must stay
exactly as shown below, in English -- {keys}. Do not translate the keys.

Here is a grounded answer to that question, already checked against NCERT textbook pages:
---
{answer}
---

There are {n} source(s), numbered [S1] to [S{n}]. Never use a higher number.

Write a four-scene script. The word budgets are hard limits, not suggestions --
each narration is a full spoken sentence, never a single word or a label --
narration is spoken aloud at about 2.3 words per second, so exceeding them
makes the video run long and the scenes desync:

{budgets}

Also give a simple concept diagram: 2 to 5 short node labels and the arrows
between them, showing how the idea flows. Node labels must be 1-4 words.
{visual_rules}
Return only valid JSON in exactly this shape:
{{
  "title": "short video title, at most 6 words",
  "scenes": {scenes_json},
  "diagram_nodes": ["...", "...", "..."],
  "diagram_edges": [["node a", "node b"], ["node b", "node c"]]
}}

"body" is what appears on screen -- keep it under 14 words, it must fit a card.
"narration" is what is spoken. They should agree but need not be identical."""


def validate(script: VideoScript, n_sources: int, require_visuals: bool = False) -> List[str]:
    """Return a list of problems. Empty means the script is usable."""
    problems: List[str] = []

    if require_visuals:
        for scene in script.scenes:
            if scene.key not in VISUAL_SCENES:
                continue
            v = scene.visual.strip()
            if not v:
                problems.append(f"scene '{scene.key}': missing 'visual' description")
            elif _DEVA.search(v):
                # The image model reads English. A Hindi prompt produces an
                # unrelated or garbled picture, so send it back.
                problems.append(f"scene '{scene.key}': 'visual' must be written in English")
            elif count_words(v) > 45:
                problems.append(f"scene '{scene.key}': 'visual' too long, keep it to one sentence")

    for scene in script.scenes:
        budget = SCENE_BUDGET.get(scene.key, (0, 25))[1]
        limit = int(budget * WORD_SLACK + 0.999)
        words = count_words(scene.narration)
        if words > limit:
            problems.append(f"scene '{scene.key}': {words} words of narration, limit {limit}")
        if not scene.narration.strip():
            problems.append(f"scene '{scene.key}': empty narration")
        elif words < SCENE_MIN_WORDS.get(scene.key, 0):
            problems.append(
                f"scene '{scene.key}': narration too short, {words} words -- write a "
                f"full spoken sentence of at least {SCENE_MIN_WORDS[scene.key]} words"
            )
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


# Question words that say nothing about the topic. Deliberately short: this
# only decides which words of the question are worth looking for.
_STOP = set("""
why what how when where which who whom whose does do did is are was were be been
the a an and or of in on at to for from with by as it its this that these those
you your they them their we our get gets got put make makes made can could will
would should there here into onto about over under between explain describe tell
कैसे क्या क्यों कब कहाँ कौन है हैं था थी थे का की के में से को और यह वह इस उस अपना अपनी
अपने होता होती होते करते करता करती बनाते बताइए समझाइए पर भी एक
कसा कसे कशी काय का कोण कुठे आहे आहेत होता होती झाला झाली झाले वर मध्ये चा ची चे
च्या ला ने नी आणि हा ही हे तो ती ते या एक सांगा
""".split())

# bge-m3 similarity above which a title is "about" the question even when it
# shares no word with it (a paraphrase, or a synonym in Hindi). Unrelated
# topics in this corpus score well below it.
TOPIC_SIM = 0.5


def _keywords(text: str) -> List[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text or ""))
            if len(w) >= 3 and w not in _STOP]


def _shares_keyword(a: List[str], b: List[str]) -> bool:
    # Four-character prefixes: "spoon"/"spoons", "पौधे"/"पौधों", "ocean"/"oceans".
    pa = {w[:4] for w in a}
    return any(w[:4] in pa for w in b)


def topic_problems(script: VideoScript, embed=None) -> List[str]:
    """Is the video about the question it was made for?

    The failure this catches was seen on Sol: a title about one thing over
    scenes about another. Word overlap decides the easy cases; the embedding
    model decides the rest, so a paraphrased title is not rejected for
    sharing no exact word with the question.
    """
    q_words = _keywords(script.question)
    if not q_words:
        return []

    narration = " ".join(s.narration for s in script.scenes)
    checks = [("title", script.title), ("narration", narration)]
    problems: List[str] = []
    for label, text in checks:
        if _shares_keyword(q_words, _keywords(text)):
            continue
        if embed is not None:
            try:
                q, t = embed([script.question, text])
                if float((q * t).sum()) >= TOPIC_SIM:
                    continue
            except Exception:  # noqa: BLE001 - a failed check must not block
                continue
        problems.append(
            f"the {label} is not about the question \"{script.question}\" -- "
            "rewrite it so it answers exactly that question"
        )
    return problems


# ---------------------------------------------------------------- grounding
#
# The script is written FROM the grounded answer, but a model rewriting 150
# words into 70 can still add a fact, swap a cause, or draw an arrow the
# answer never states. A video is more persuasive than text, so each factual
# line and every diagram arrow is checked against the answer before render.
#
# The model is the reviewer -- it can tell a faithful paraphrase from a new
# claim, which word overlap cannot. Embeddings are the second opinion: a line
# that plainly restates an answer sentence is not rejected on a 7B judge's
# whim, and with no model at all they are the whole check.

FACT_SCENES = ("idea", "diagram")      # title is a label; check is a question
SUPPORT_SIM = 0.55                     # a claim this close to an answer sentence is supported
STRONG_SIM = 0.75                      # ...this close overrides the judge
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _claims(script: VideoScript) -> List[Dict[str, str]]:
    out = []
    for s in script.scenes:
        if s.key in FACT_SCENES and s.narration.strip():
            out.append({"id": f"C{len(out) + 1}", "kind": "scene", "scene": s.key, "text": s.narration})
    for a, b in script.diagram_edges:
        out.append({"id": f"D{len(out) + 1}", "kind": "arrow", "scene": "diagram",
                    "text": f"{a} -> {b}", "edge": f"{a} -> {b}"})
    return out


def _judge_prompt(evidence: str, claims: List[Dict[str, str]]) -> str:
    listed = "\n".join(f"[{c['id']}] {c['text']}" for c in claims)
    return f"""You check a short school video script against its source.

EVIDENCE -- an answer already verified against NCERT textbook pages:
---
{evidence}
---

CLAIMS from the video script. Lines marked D are diagram arrows: "A -> B"
means the video shows A leading to, causing or feeding into B.
{listed}

For each claim decide: does the EVIDENCE support it? A faithful paraphrase,
a simplification, or a translation of the evidence is supported. An everyday
example is fine if it states no new fact. A claim is UNSUPPORTED if it adds
a fact, number, name or cause the evidence does not state, reverses a
relationship, or contradicts the evidence.

Return only valid JSON:
{{"unsupported": [{{"id": "C1", "why": "one short reason"}}]}}
Use an empty list when every claim is supported."""


def _answer_sentences(evidence: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(evidence or "") if len(s.strip()) > 8]


def _best_support(embed, claim: str, sentences: List[str]) -> Optional[float]:
    if embed is None or not sentences:
        return None
    try:
        vecs = embed([claim] + sentences)
        return float(max((vecs[0] * v).sum() for v in vecs[1:]))
    except Exception:  # noqa: BLE001
        return None


def grounding_problems(script: VideoScript, evidence: str, providers=None, embed=None):
    """(problems, report). Problems are fed back for a rewrite; the report
    goes into the video's audit file."""
    claims = _claims(script)
    sentences = _answer_sentences(evidence)
    report: Dict[str, Any] = {"checked": len(claims), "method": None, "unsupported": []}
    if not claims or not evidence.strip():
        report["method"] = "skipped"
        return [], report

    flagged: Dict[str, str] = {}
    judged = False
    if providers:
        from story_mvp.model_clients import parse_json_response
        from story_mvp.rag_chat import _call_provider

        for provider in providers:
            try:
                raw = _call_provider(provider, _judge_prompt(evidence, claims),
                                     "Check the claims now. Reply with the JSON only.", None)
                data = parse_json_response(raw)
                for item in (data.get("unsupported") or []) if isinstance(data, dict) else []:
                    if isinstance(item, dict) and item.get("id"):
                        flagged[str(item["id"]).strip("[] ")] = str(item.get("why", "")).strip()
                    elif isinstance(item, str):
                        flagged[item.strip("[] ")] = ""
                judged = True
                break
            except Exception as exc:  # noqa: BLE001 - fall back to embeddings
                print(f"grounding judge: {getattr(provider, 'provider_name', '?')} failed - {exc}")

    problems: List[str] = []
    for c in claims:
        sim = _best_support(embed, c["text"], sentences)
        if judged:
            unsupported = c["id"] in flagged and not (sim is not None and sim >= STRONG_SIM)
            why = flagged.get(c["id"], "")
        elif sim is not None:
            unsupported, why = sim < SUPPORT_SIM, f"closest answer sentence scores {sim:.2f}"
        elif c["kind"] == "arrow":
            # No model, no embeddings: both ends of an arrow must at least be
            # things the answer talks about.
            ends = c["edge"].split(" -> ")
            unsupported = not all(_shares_keyword(_keywords(evidence), _keywords(e)) for e in ends)
            why = "the answer never mentions one end of this arrow"
        else:
            continue
        if not unsupported:
            continue
        report["unsupported"].append({"id": c["id"], "text": c["text"], "why": why,
                                      "similarity": None if sim is None else round(sim, 3)})
        if c["kind"] == "arrow":
            problems.append(f"diagram arrow '{c['edge']}' is not supported by the answer"
                            f"{' (' + why + ')' if why else ''} -- use only relationships the answer states")
        else:
            problems.append(f"scene '{c['scene']}' narration says something the answer does not support"
                            f"{' (' + why + ')' if why else ''} -- use only facts from the answer")

    report["method"] = "model_judge+embedding" if judged else ("embedding" if embed else "keyword")
    report["verified"] = not problems
    return problems, report


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
    want_visuals: bool = False,
    embed=None,
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
        rag_result.get("question", ""), rag_result.get("answer", ""), sources, request,
        want_visuals=want_visuals,
    )
    client = getattr(llm_client, "client", llm_client)
    providers = list(getattr(client, "clients", [client]))

    last_problems: List[str] = []
    # A script that is correct but a little short, or whose picture
    # descriptions are unusable, is still a video -- those scenes get text
    # cards. The fullest such script is kept so these never cost the video;
    # the retries exist to do better, not to refuse.
    fallback: Optional[VideoScript] = None
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
        problems = validate(script, len(sources), require_visuals=want_visuals)
        problems += topic_problems(script, embed)
        # Fact-check last: it costs a model call, so only a script that is
        # otherwise usable is checked. Soft problems (short, bad visual) do
        # not skip it -- the fallback script must be verified too.
        if all(_is_soft(p) for p in problems):
            g_problems, script.grounding = grounding_problems(
                script, _evidence(rag_result), providers, embed)
            problems += g_problems
            print(f"video script attempt {attempt + 1}: {script.grounding['checked']} claims checked "
                  f"({script.grounding['method']}), {len(g_problems)} unsupported")
        if not problems:
            return script
        if all(_is_soft(p) for p in problems) and (
                fallback is None or script.total_words > fallback.total_words):
            fallback = _drop_bad_visuals(script)
        last_problems = problems
        print(f"video script attempt {attempt+1} rejected: {problems}")
        _dump_reply(debug_dir, request, attempt + 1, raw, problems)

    if fallback is not None:
        print(f"video script: keeping the best attempt ({fallback.total_words} words); "
              f"remaining issues: {last_problems}")
        return fallback
    raise ScriptError(f"could not produce a valid script in {max_attempts} attempts: {last_problems}")


def _evidence(rag_result: Dict[str, Any]) -> str:
    """What the script may state: the grounded answer, plus any source
    excerpts that came with it."""
    parts = [str(rag_result.get("answer") or "")]
    parts += [str(s.get("excerpt")) for s in (rag_result.get("sources") or [])
              if isinstance(s, dict) and s.get("excerpt")]
    return "\n".join(p for p in parts if p.strip())


def _is_soft(problem: str) -> bool:
    """Problems that make a video weaker, not wrong."""
    return "'visual'" in problem or "too short" in problem


def _drop_bad_visuals(script: VideoScript) -> VideoScript:
    for scene in script.scenes:
        v = scene.visual.strip()
        if _DEVA.search(v) or count_words(v) > 45:
            scene.visual = ""
    return script


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


_SCENE_FIELDS = ("heading", "body", "narration", "visual")


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
            visual=str(s.get("visual", "")).strip(),
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
