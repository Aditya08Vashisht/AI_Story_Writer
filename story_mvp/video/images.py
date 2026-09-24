"""Scene illustrations with FLUX.1-schnell (Apache-2.0).

Vidplan.md first ruled out any image model, for two reasons: diffusion mangles
on-screen text, and it gives a different result every run. Both are solved here
without giving up pictures:

  text           the model NEVER draws text. Prompts describe objects and
                 scenes only, the style suffix forbids lettering, and every
                 word on screen is still drawn by Pillow.
  reproducible   each image uses a seed derived from its prompt, and results
                 are cached on disk by (model, prompt, seed, size, steps). The
                 same script gives the same images, and a re-run costs nothing.

Illustrations are atmosphere and example, never evidence. A picture of a globe
may get a coastline wrong, so no fact is ever carried by an image -- facts live
in the cited narration and the Pillow-drawn diagram, where they can be checked.

Optional by design: without a GPU or diffusers, videos fall back to text cards
rather than failing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

MODEL_ID = os.environ.get("STORYTUTOR_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

# 16:9, both multiples of 16 as FLUX requires. Upscaled to 1280x720 on the card.
GEN_W, GEN_H = 1024, 576

# schnell is distilled for 1-4 steps; more buys nothing.
STEPS = int(os.environ.get("STORYTUTOR_IMAGE_STEPS", "4"))

# One style for every scene, so the three pictures in a video look like they
# belong together rather than like three unrelated stock images.
STYLE = (
    "flat vector illustration in the style of a modern school textbook, soft "
    "friendly colours, clean simple shapes, gentle lighting, uncluttered "
    "background, no text, no letters, no words, no numbers, no labels, "
    "no signs, no captions, no watermark"
)


def cache_dir() -> Path:
    return Path(os.environ.get("STORYTUTOR_IMAGE_CACHE", "outputs/illustration_cache"))


def full_prompt(visual: str) -> str:
    return f"{(visual or '').strip().rstrip('.')}. {STYLE}"


def seed_for(prompt: str) -> int:
    """Deterministic per prompt, so one scene always gets the same image."""
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


class IllustrationGenerator:
    """Lazily loaded: importing the video package never pulls in diffusers."""

    def __init__(self, model_id: str = None, steps: int = None):
        self.model_id = model_id or MODEL_ID
        self.steps = steps or STEPS
        self._pipe = None
        self._failed: Optional[str] = None

    def cache_path(self, visual: str) -> Tuple[Path, str, int]:
        prompt = full_prompt(visual)
        seed = seed_for(prompt)
        key = hashlib.sha256(
            f"{self.model_id}|{prompt}|{seed}|{GEN_W}x{GEN_H}|{self.steps}".encode("utf-8")
        ).hexdigest()[:20]
        return cache_dir() / f"{key}.png", prompt, seed

    def _load(self) -> None:
        if self._pipe is not None or self._failed:
            return
        try:
            import torch
            from diffusers import FluxPipeline

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "no CUDA device -- FLUX on CPU takes tens of minutes per image"
                )
            print(f"loading image model {self.model_id} "
                  "(first run downloads ~33 GB into HF_HOME) ...")
            pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
            # Everything resident is fastest and fits an 80 GB A100 alongside
            # the retrieval models and Ollama. Offload trades speed for memory
            # on a smaller card or a crowded one.
            if os.environ.get("STORYTUTOR_IMAGE_OFFLOAD", "").lower() in {"1", "true", "yes"}:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            self._pipe = pipe
        except Exception as exc:  # noqa: BLE001
            self._failed = str(exc)
            print(f"illustrations unavailable, falling back to text cards - {exc}")

    @property
    def available(self) -> bool:
        self._load()
        return self._pipe is not None

    @property
    def failure(self) -> Optional[str]:
        return self._failed

    def generate(self, visual: str):
        """Return (PIL.Image, meta) or None. Cached first, generated second."""
        if not (visual or "").strip():
            return None
        path, prompt, seed = self.cache_path(visual)
        meta = {"visual": visual, "prompt": prompt, "seed": seed,
                "model": self.model_id, "steps": self.steps, "file": str(path)}

        if path.exists():
            from PIL import Image

            return Image.open(path).convert("RGB"), {**meta, "cached": True}

        self._load()
        if self._pipe is None:
            return None
        try:
            import torch

            # A CPU generator makes the seed reproducible across GPUs and
            # driver versions, which a CUDA generator does not guarantee.
            gen = torch.Generator("cpu").manual_seed(seed)
            image = self._pipe(
                prompt,
                width=GEN_W,
                height=GEN_H,
                num_inference_steps=self.steps,
                guidance_scale=0.0,            # schnell is guidance-distilled
                max_sequence_length=256,       # schnell's maximum
                generator=gen,
            ).images[0]
        except Exception as exc:  # noqa: BLE001
            print(f"illustration failed for one scene, using a text card - {exc}")
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return image.convert("RGB"), {**meta, "cached": False}
