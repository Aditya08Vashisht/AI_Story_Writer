"""Assemble rendered cards (and optional narration) into an MP4.

Uses imageio-ffmpeg, which ships a static ffmpeg binary as a pip wheel. That
avoids `module load ffmpeg` entirely -- on this cluster every failure so far has
been an install problem, not a code problem, so the fewer external binaries the
better.

Narration is optional by design. Without TTS you still get a watchable video
with burned subtitles, which is enough to demonstrate the pipeline; with TTS the
real audio duration drives each scene's length instead of the word-count
estimate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

FPS = 30
W, H = 1280, 720


class RenderError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    """Prefer the pip-installed static binary; fall back to a system one."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise RenderError(
        "No ffmpeg available. Install the bundled one with:  pip install imageio-ffmpeg"
    )


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(scenes: Sequence, durations: Sequence[float], path: Path) -> Path:
    lines, t = [], 0.0
    for i, (scene, dur) in enumerate(zip(scenes, durations), 1):
        text = (scene.narration or scene.body or "").strip()
        if text:
            lines.append(str(i))
            lines.append(f"{_srt_timestamp(t)} --> {_srt_timestamp(t + dur)}")
            lines.append(text)
            lines.append("")
        t += dur
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_video(
    script,
    images: Sequence,
    audio_paths: Optional[Sequence[Optional[Path]]] = None,
    durations: Optional[Sequence[float]] = None,
    out_path: Path = Path("outputs/videos/video.mp4"),
    burn_subtitles: bool = False,
    motion: bool = True,
) -> Path:
    """Cards + optional narration -> MP4."""
    if not images:
        raise RenderError("no frames to render")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_binary()

    if durations is None:
        durations = [s.seconds for s in script.scenes]
    durations = [max(1.5, float(d)) for d in durations]

    work = Path(tempfile.mkdtemp(prefix="stvid_"))
    try:
        # 1. One short clip per scene: a slow push-in plus a fade in and out,
        #    so each card moves instead of sitting as a frozen slide.
        clips = []
        for i, img in enumerate(images):
            png = work / f"scene_{i:02d}.png"
            img.save(png)
            clip = work / f"scene_{i:02d}.mp4"
            _scene_clip(ffmpeg, png, durations[i], clip, motion=motion)
            clips.append(clip)

        listing = work / "clips.txt"
        listing.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8")

        # 2. Join the clips, burning subtitles in the same pass.
        silent = work / "silent.mp4"
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
        if burn_subtitles:
            srt = write_srt(script.scenes, durations, work / "subs.srt")
            cmd += ["-vf", _subtitle_filter(srt, script)]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(FPS), str(silent)]
        _run(cmd)

        real_audio = [p for p in (audio_paths or []) if p and Path(p).exists()]
        if not real_audio:
            shutil.move(str(silent), str(out_path))
            return out_path

        # Concatenate per-scene narration, then mux.
        alist = work / "audio.txt"
        alist.write_text(
            "\n".join(f"file '{Path(p).as_posix()}'" for p in audio_paths if p and Path(p).exists()),
            encoding="utf-8",
        )
        merged = work / "narration.wav"
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(alist),
              "-c", "copy", str(merged)])
        _run([ffmpeg, "-y", "-i", str(silent), "-i", str(merged),
              "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_path)])
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _escape_filter_path(p) -> str:
    # ffmpeg filter arguments treat ':' as a separator, including the one in a
    # Windows drive letter, and want forward slashes.
    return str(p).replace("\\", "/").replace(":", "\\:")


def _scene_clip(ffmpeg: str, png: Path, seconds: float, out: Path, motion: bool = True) -> None:
    frames = max(1, int(round(seconds * FPS)))
    fade = min(0.45, seconds / 5)
    if motion:
        # A 5% push-in over the scene. The source is upscaled first because
        # zoompan steps in whole pixels and visibly jitters at 1280 wide.
        step = 0.05 / frames
        vf = [
            "scale=2560:-2",
            f"zoompan=z='min(zoom+{step:.6f},1.05)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS}",
        ]
        head = [ffmpeg, "-y", "-i", str(png)]
        tail = ["-frames:v", str(frames)]
    else:
        vf = [f"scale={W}:{H}", f"fps={FPS}"]
        head = [ffmpeg, "-y", "-loop", "1", "-i", str(png)]
        tail = ["-t", f"{seconds:.3f}"]
    vf += [f"fade=t=in:st=0:d={fade:.2f}",
           f"fade=t=out:st={max(0.0, seconds - fade):.2f}:d={fade:.2f}",
           "format=yuv420p"]
    _run(head + ["-vf", ",".join(vf)] + tail +
         ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS), str(out)])


def _subtitle_filter(srt: Path, script) -> str:
    """Burn subtitles in a font that can actually render the script.

    libass falls back to system fonts through fontconfig, and a cluster node
    usually has none with Devanagari -- Hindi and Marathi subtitles would come
    out as boxes. Pointing it at the downloaded Noto fonts avoids that. A
    translucent box behind the text keeps it readable over an illustration.
    """
    from story_mvp.video.cards import FONT_DIR, has_devanagari

    deva = any(has_devanagari(s.narration) for s in script.scenes)
    family = "Noto Sans Devanagari" if deva else "Noto Sans"
    style = (f"FontName={family},FontSize=15,PrimaryColour=&H00F6EDE6,"
             "BorderStyle=3,BackColour=&H99000000,Outline=1,Shadow=0,MarginV=18")
    f = f"subtitles='{_escape_filter_path(srt)}'"
    if Path(FONT_DIR).exists():
        f += f":fontsdir='{_escape_filter_path(FONT_DIR)}'"
    return f + f":force_style='{style}'"


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1200:]
        raise RenderError(f"ffmpeg failed ({proc.returncode}):\n{tail}")


def write_sidecar(script, video_path: Path, extra: Optional[dict] = None) -> Path:
    """Audit trail: question, sources with scores, script, model.

    Without this a video is an unfalsifiable claim. With it, a reviewer can
    check every line against the page it came from.
    """
    data = script.to_dict()
    data["video"] = str(video_path)
    if extra:
        data.update(extra)
    side = Path(video_path).with_suffix(".json")
    side.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return side
