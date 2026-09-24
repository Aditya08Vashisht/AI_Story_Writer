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
        # One still per scene, held for its duration. concat demuxer needs the
        # final entry repeated or it drops the last frame.
        entries, png_paths = [], []
        for i, img in enumerate(images):
            p = work / f"scene_{i:02d}.png"
            img.save(p)
            png_paths.append(p)
            entries.append(f"file '{p.as_posix()}'")
            entries.append(f"duration {durations[i]:.3f}")
        entries.append(f"file '{png_paths[-1].as_posix()}'")
        concat = work / "concat.txt"
        concat.write_text("\n".join(entries), encoding="utf-8")

        silent = work / "silent.mp4"
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat)]

        vf = [f"scale={W}:{H}:force_original_aspect_ratio=decrease",
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0b0f17",
              f"fps={FPS}", "format=yuv420p"]
        if burn_subtitles:
            srt = write_srt(script.scenes, durations, work / "subs.srt")
            # ffmpeg's subtitles filter needs escaped path separators
            esc = str(srt).replace("\\", "/").replace(":", "\\:")
            vf.insert(2, f"subtitles='{esc}':force_style='FontSize=20,PrimaryColour=&H00E6EDF6'")

        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-r", str(FPS), str(silent)]
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
