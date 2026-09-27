"""One grounded answer -> one 30-second video. Shared by the CLI and the chat.

scripts/make_video.py runs it for a batch; the chat page's "Make video"
button (app.py /api/video) runs it for the answer on screen, reusing that
exact answer and its sources so the video explains what the student read.
"""

from __future__ import annotations

import hashlib
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Optional

from story_mvp.video import cards, render as rnd
from story_mvp.video.script import VISUAL_SCENES, script_from_answer
from story_mvp.video.tts import audio_duration


def make_embedder(engine):
    """bge-m3 from the retrieval engine, for the script's topic check."""
    model = getattr(engine, "model", None)
    if model is None or not hasattr(model, "encode"):
        return None
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def slugify(text: str, limit: int = 48) -> str:
    """Readable and unique. isalnum() drops Devanagari vowel signs, so two
    Hindi questions could collapse to one name and overwrite each other's
    video -- the title of one over the scenes of another. The hash makes
    every question its own file."""
    keep = "".join(c if c.isalnum() or c in " -_" or unicodedata.category(c).startswith("M")
                   else "" for c in text)
    stem = "-".join(keep.lower().split())[:limit] or "video"
    return f"{stem}-{hashlib.sha1(text.strip().encode('utf-8')).hexdigest()[:6]}"


def load_illustrator():
    """(generator, None) when pictures are available, else (None, reason)."""
    from story_mvp.video.images import IllustrationGenerator

    gen = IllustrationGenerator()
    if gen.available:
        return gen, None
    reason = gen.failure or "unknown error"
    if "gated" in reason or "403" in reason or "401" in reason:
        reason = (f"the image model is gated -- signed in to Hugging Face, open "
                  f"https://huggingface.co/{gen.model_id} and click 'Agree and access repository'")
    elif "No module" in reason or "cannot import" in reason:
        reason = "pip install -U diffusers accelerate sentencepiece protobuf"
    return None, reason.splitlines()[0]


def make_video(concept: Dict, engine, llm, tts, outdir: Path, illustrator=None,
               rag: Optional[Dict] = None,
               progress: Callable[[str, float], None] = None) -> Dict:
    """Build one video. `rag` reuses an answer already shown in the chat;
    without it the question is answered afresh. `progress(step, fraction)`
    reports each stage for the chat page's progress bar."""
    say = progress or (lambda step, frac: None)
    outdir = Path(outdir)
    question = concept["question"]
    request = {
        "question": question,
        "language": concept.get("language", "english"),
        "class_level": str(concept.get("class_level", "")),
        "subject": concept.get("subject", ""),
    }
    print(f"\n{'=' * 70}\n{question}\n  {request['language']} · class "
          f"{request['class_level'] or '-'} · {request['subject'] or '-'}")

    t0 = time.time()
    if rag is None:
        from story_mvp.rag_chat import answer_question

        say("Finding the textbook pages", 0.05)
        rag = answer_question(request, engine, llm)
    rag = {**rag, "question": question}
    print(f"  retrieval: {len(rag.get('sources') or [])} sources, grounded={rag.get('grounded')}")
    if not rag.get("grounded") or not rag.get("sources"):
        print("  SKIPPED: no grounded sources.")
        return {"question": question, "status": "refused_ungrounded",
                "error": "This answer has no textbook sources, so no video is made for it."}

    say("Writing the 30-second script", 0.15)
    t_script = time.time()
    script = script_from_answer(rag, llm, request, debug_dir=outdir / "debug",
                                want_visuals=illustrator is not None, embed=make_embedder(engine))
    print(f"  script: {script.total_words} words across {len(script.scenes)} scenes "
          f"({time.time() - t_script:.0f}s)")

    # Illustrations: one per title/idea/check scene. One failure gives that
    # scene a text card; it never costs the video.
    pictures, illustration_log = {}, []
    if illustrator is not None:
        todo = [s for s in script.scenes if s.key in VISUAL_SCENES and s.visual]
        for n, scene in enumerate(todo):
            say(f"Painting picture {n + 1} of {len(todo)}", 0.3 + 0.3 * n / max(1, len(todo)))
            result = illustrator.generate(scene.visual)
            if result is None:
                illustration_log.append({"scene": scene.key, "visual": scene.visual, "generated": False})
                continue
            img, meta = result
            pictures[scene.key] = img
            illustration_log.append({"scene": scene.key, **meta, "generated": True})
            print(f"    image '{scene.key}': {'cached' if meta['cached'] else 'generated'} "
                  f"(seed {meta['seed']}) - {scene.visual[:60]}")

    try:
        images = [cards.render_scene(s, script, pictures.get(s.key)) for s in script.scenes]
    except cards.RenderError as exc:
        print(f"  RENDER REFUSED: {exc}")
        return {"question": question, "status": "refused_no_shaper", "error": str(exc)}

    # Narrate. Real audio length drives each scene's duration.
    audio_paths, durations = [], []
    stem = slugify(question)
    for i, scene in enumerate(script.scenes):
        path = None
        if tts is not None:
            say(f"Recording the voice, line {i + 1} of {len(script.scenes)}",
                0.62 + 0.2 * i / len(script.scenes))
            path = tts.speak(scene.narration, script.language, outdir / "audio" / f"{stem}_{i}.wav")
        audio_paths.append(path)
        dur = audio_duration(path) if path else None
        durations.append(max(2.0, dur + 0.6) if dur else scene.seconds)
    narrated = sum(1 for p in audio_paths if p)
    if tts is not None and narrated < len(script.scenes):
        print(f"  WARNING: only {narrated}/{len(script.scenes)} lines were voiced -- see the voice server log")

    say("Putting the video together", 0.85)
    video = rnd.render_video(script, images, audio_paths, durations, outdir / f"{stem}.mp4",
                             burn_subtitles=True)
    side = rnd.write_sidecar(script, video, {
        "rag_answer": rag.get("answer"),
        "retrieval_method": rag.get("retrieval_method"),
        "seconds_to_build": round(time.time() - t0, 1),
        "narrated": narrated > 0,
        "voice": (getattr(tts, "engine", None) or getattr(tts, "name", None)) if tts else None,
        "illustrations": illustration_log,
        "image_model": getattr(illustrator, "model_id", None),
    })
    total = sum(durations)
    print(f"  wrote {video}  ({total:.1f}s, narrated {narrated}/{len(script.scenes)} lines)")
    print(f"  audit {side}")
    say("Done", 1.0)
    return {"question": question, "status": "ok", "video": str(video), "file": Path(video).name,
            "title": script.title, "seconds": round(total, 1), "narrated": narrated > 0,
            "illustrations": len(pictures), "sources": len(rag["sources"]),
            "verified": script.grounding.get("verified")}
