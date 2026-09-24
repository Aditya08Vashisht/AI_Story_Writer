"""Make a 30-second concept video from a grounded RAG answer.

  python scripts/make_video.py --question "Why does a spoon get hot in hot water?" \
      --class-level 6 --subject science --language english

  python scripts/make_video.py --batch concepts.json

Requires the retrieval index, and Ollama (or another provider) for the script.
Refuses to make a video for a question retrieval cannot ground -- see Vidplan.md
section 5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.rag_chat import answer_question  # noqa: E402
from story_mvp.video import cards, render as rnd  # noqa: E402
from story_mvp.video.script import ScriptError, script_from_answer  # noqa: E402
from story_mvp.video.tts import IndicTTS, audio_duration  # noqa: E402


def preflight_llm() -> bool:
    """Confirm a generation provider answers before anything expensive loads."""
    import urllib.request

    from story_mvp.model_clients import normalize_ollama_host

    # Same normalisation the client uses. A private copy here is how preflight
    # passed while every real request failed on the scheme-less host.
    host = normalize_ollama_host(os.environ.get("OLLAMA_HOST", ""))
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"\nNo model server at {host} ({exc}).\n")
        print("Start Ollama first:\n")
        print("  export PATH=/scratch/$USER/ollama/bin:$PATH")
        print("  export LD_LIBRARY_PATH=/scratch/$USER/ollama/lib/ollama:$LD_LIBRARY_PATH")
        print("  export OLLAMA_MODELS=/scratch/$USER/ollama/models")
        print("  export OLLAMA_HOST=127.0.0.1:11434")
        print("  nohup ollama serve > /scratch/$USER/ollama/serve.log 2>&1 &")
        print("  sleep 6 && curl -s http://127.0.0.1:11434/api/tags\n")
        return False

    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n == model or n.startswith(model.split(":")[0]) for n in names):
        print(f"\n{host} is up, but '{model}' is not pulled. Available: {names or 'none'}")
        print(f"  ollama pull {model}\n")
        return False

    print(f"model server OK at {host} · {model}")
    return True


def slugify(text: str, limit: int = 48) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in text)
    return "-".join(keep.lower().split())[:limit] or "video"


def make_one(concept: dict, engine, llm, tts, outdir: Path, no_audio: bool = False,
             illustrator=None) -> dict:
    question = concept["question"]
    request = {
        "question": question,
        "language": concept.get("language", "english"),
        "class_level": str(concept.get("class_level", "")),
        "subject": concept.get("subject", ""),
    }
    print(f"\n{'='*70}\n{question}\n  {request['language']} · class {request['class_level'] or '-'} · {request['subject'] or '-'}")

    t0 = time.time()
    rag = answer_question(request, engine, llm)
    rag["question"] = question
    print(f"  retrieval: {len(rag.get('sources') or [])} sources, grounded={rag.get('grounded')}")

    if not rag.get("grounded") or not rag.get("sources"):
        print("  SKIPPED: no grounded sources. Refusing to make a confident video "
              "about something the textbooks do not cover.")
        return {"question": question, "status": "refused_ungrounded"}

    t_script = time.time()
    want_visuals = illustrator is not None
    script = script_from_answer(rag, llm, request, debug_dir=outdir / "debug",
                                want_visuals=want_visuals)
    print(f"  script: {script.total_words} words across {len(script.scenes)} scenes "
          f"({time.time() - t_script:.0f}s)")

    # Illustrations: one per title/idea/check scene. Any single failure falls
    # back to a text card for that scene rather than losing the video.
    pictures, illustration_log = {}, []
    if illustrator is not None:
        t_img = time.time()
        from story_mvp.video.script import VISUAL_SCENES

        for scene in script.scenes:
            if scene.key not in VISUAL_SCENES or not scene.visual:
                continue
            result = illustrator.generate(scene.visual)
            if result is None:
                illustration_log.append({"scene": scene.key, "visual": scene.visual, "generated": False})
                continue
            img, meta = result
            pictures[scene.key] = img
            illustration_log.append({"scene": scene.key, **meta, "generated": True})
            print(f"    image '{scene.key}': {'cached' if meta['cached'] else 'generated'} "
                  f"(seed {meta['seed']}) - {scene.visual[:60]}")
        print(f"  illustrations: {len(pictures)}/{len(VISUAL_SCENES)} ({time.time() - t_img:.0f}s)")

    # Render cards. Refuse Devanagari if this Pillow cannot shape it.
    try:
        images = [cards.render_scene(s, script, pictures.get(s.key)) for s in script.scenes]
    except cards.RenderError as exc:
        print(f"  RENDER REFUSED: {exc}")
        return {"question": question, "status": "refused_no_shaper", "error": str(exc)}

    # Narrate. Real audio length drives scene duration when available.
    audio_paths, durations = [], []
    work = outdir / "audio"
    for i, scene in enumerate(script.scenes):
        path = None
        if tts is not None and not no_audio:
            path = tts.speak(scene.narration, script.language, work / f"{slugify(question)}_{i}.wav")
        audio_paths.append(path)
        dur = audio_duration(path) if path else None
        durations.append((dur + 0.6) if dur else scene.seconds)

    out = outdir / f"{slugify(question)}.mp4"
    video = rnd.render_video(script, images, audio_paths, durations, out, burn_subtitles=True)
    side = rnd.write_sidecar(script, video, {
        "rag_answer": rag.get("answer"),
        "retrieval_method": rag.get("retrieval_method"),
        "seconds_to_build": round(time.time() - t0, 1),
        "narrated": any(p for p in audio_paths),
        "illustrations": illustration_log,
        "image_model": getattr(illustrator, "model_id", None),
    })

    total = sum(durations)
    print(f"  wrote {video}  ({total:.1f}s, narrated={any(audio_paths)})")
    print(f"  audit {side}")
    return {"question": question, "status": "ok", "video": str(video),
            "seconds": round(total, 1), "sources": len(rag["sources"])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a 30-second concept video.")
    ap.add_argument("--question")
    ap.add_argument("--language", default="english", choices=["english", "hindi", "marathi"])
    ap.add_argument("--class-level", default="")
    ap.add_argument("--subject", default="", choices=["", "science", "social_science"])
    ap.add_argument("--batch", help="JSON file: a list of concept objects")
    ap.add_argument("--outdir", default="outputs/videos")
    ap.add_argument("--dataset-dir", default="knowledge_base")
    ap.add_argument("--no-audio", action="store_true", help="Skip TTS; subtitles only.")
    ap.add_argument("--no-images", action="store_true",
                    help="Skip illustrations; text cards only (no GPU image model).")
    args = ap.parse_args()

    if not args.question and not args.batch:
        ap.error("give --question or --batch")

    concepts = (json.loads(Path(args.batch).read_text(encoding="utf-8")) if args.batch
                else [{"question": args.question, "language": args.language,
                       "class_level": args.class_level, "subject": args.subject}])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check the model BEFORE loading the embedding model and reranker. Those
    # take a minute on GPU, and discovering "connection refused" afterwards
    # wastes all of it -- and the failure then looks like a video problem
    # rather than a server that was never started.
    if not preflight_llm():
        return 2

    from story_mvp.llm_client import StoryLLM
    from story_mvp.rag_engine import RetrievalService

    engine = RetrievalService(dataset_dir=args.dataset_dir)
    llm = StoryLLM()
    tts = None if args.no_audio else IndicTTS()

    illustrator = None
    if not args.no_images:
        from story_mvp.video.images import IllustrationGenerator

        illustrator = IllustrationGenerator()
        # Load once, up front, so a missing diffusers or GPU is reported
        # before any concept runs -- and then the batch continues on text cards.
        if not illustrator.available:
            reason = illustrator.failure or ""
            print("\nNOTE: illustrations disabled -- videos will use text cards.")
            if "gated" in reason or "403" in reason or "401" in reason:
                print("      The image model is gated. Signed in to Hugging Face, open")
                print(f"        https://huggingface.co/{illustrator.model_id}")
                print("      click 'Agree and access repository', then run again.\n")
            elif "No module" in reason or "cannot import" in reason:
                print("      pip install -U diffusers accelerate sentencepiece protobuf\n")
            else:
                print(f"      {reason.splitlines()[0] if reason else 'unknown error'}\n")
            illustrator = None

    results = []
    for concept in concepts:
        try:
            results.append(make_one(concept, engine, llm, tts, outdir, args.no_audio,
                                    illustrator=illustrator))
        except ScriptError as exc:
            print(f"  SCRIPT REFUSED: {exc}")
            results.append({"question": concept.get("question"), "status": "refused_script",
                            "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            results.append({"question": concept.get("question"), "status": "error", "error": str(exc)})

    print("\n" + "=" * 70)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"{ok}/{len(results)} videos rendered")
    for r in results:
        mark = "OK " if r["status"] == "ok" else "-- "
        print(f"  {mark} {r['status']:<22} {str(r.get('question'))[:60]}")

    (outdir / "batch_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
