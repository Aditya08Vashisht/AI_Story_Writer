"""Speech for the chatbot answers and the video narration.

Two engines, both free:

  parler  ai4bharat/indic-parler-tts (Apache-2.0). Natural Hindi, Marathi and
          Indian English from one model. The preferred voice.
  mms     facebook/mms-tts-{hin,mar,eng} (CC-BY-NC-4.0, fine for research).
          Plainer, but small, fast and loads with stock transformers. The
          fallback if Parler cannot load.

Why a separate server (scripts/tts_server.py) rather than loading here:
`parler-tts` pins an old transformers, and the main environment needs a new
one for diffusers and bge-m3. Installing both in one environment breaks one of
them. So speech runs in its own environment, and the app and the video maker
talk to it over HTTP (RemoteTTS). Everything degrades gracefully: no server
means silent videos with subtitles, and the chat page falls back to the
browser's own voice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import List, Optional

MODEL_ID = os.environ.get("STORYTUTOR_TTS_MODEL", "ai4bharat/indic-parler-tts")
TTS_URL = os.environ.get("STORYTUTOR_TTS_URL", "http://127.0.0.1:5060")

# Parler is steered by a text description of the voice. The description goes
# through an English text encoder (flan-t5), so it must be written in English
# even for a Hindi voice -- a Hindi description is noise to it. The named
# speakers are the ones the model card recommends for each language.
VOICE = {
    "english": "Mary speaks at a moderate pace with a clear, warm and friendly tone. "
               "The recording is of very high quality, close up, with no background noise.",
    "hindi": "Divya speaks at a moderate pace with a clear, warm and expressive tone. "
             "The recording is of very high quality, close up, with no background noise.",
    "marathi": "Sunita speaks at a moderate pace with a clear, warm and expressive tone. "
               "The recording is of very high quality, close up, with no background noise.",
}

MMS_CODE = {"english": "eng", "hindi": "hin", "marathi": "mar"}

_CITE = re.compile(r"\[S\d+\]")
_MD = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.M)
_URL = re.compile(r"https?://\S+")
_SENT = re.compile(r"(?<=[.!?।॥])\s+")


class TTSUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- text prep

def clean_for_speech(text: str) -> str:
    """What a listener should hear: no [S1], no markdown, no URLs."""
    t = _CITE.sub("", text or "")
    t = _URL.sub("", t)
    t = _MD.sub("", t)
    t = re.sub(r"[()\[\]{}<>|~^=_*#]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_sentences(text: str, max_chars: int = 220) -> List[str]:
    """Sentence-sized pieces. Parler degrades on long inputs, and short pieces
    let the chat page start speaking after the first one instead of the last."""
    pieces: List[str] = []
    for sent in _SENT.split(clean_for_speech(text)):
        sent = sent.strip()
        while len(sent) > max_chars:
            cut = max(sent.rfind(",", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > max_chars // 3 else max_chars
            pieces.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        if sent:
            pieces.append(sent)
    # Fold very short fragments into the previous piece: "Yes." alone sounds clipped.
    merged: List[str] = []
    for p in pieces:
        if merged and len(p) < 25 and len(merged[-1]) + len(p) < max_chars:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def cache_key(text: str, language: str, engine: str) -> str:
    return hashlib.sha256(f"{engine}|{language}|{text}".encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------- engines

class _Engine:
    name = "none"

    def __init__(self):
        self._failed: Optional[str] = None
        self.sample_rate = 16000

    @property
    def available(self) -> bool:
        self._load()
        return self._failed is None

    @property
    def failure(self) -> Optional[str]:
        return self._failed

    def _load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def synth(self, text: str, language: str):  # pragma: no cover - overridden
        raise NotImplementedError

    def speak_array(self, text: str, language: str):
        """Whole text -> one float32 array, sentence by sentence with pauses."""
        import numpy as np

        self._load()
        if self._failed:
            raise TTSUnavailable(self._failed)
        gap = np.zeros(int(self.sample_rate * 0.25), dtype=np.float32)
        parts = []
        for sent in split_sentences(text):
            parts += [np.asarray(self.synth(sent, language), dtype=np.float32).reshape(-1), gap]
        if not parts:
            raise TTSUnavailable("nothing to say")
        return np.concatenate(parts[:-1])

    def speak(self, text: str, language: str, out_path: Path) -> Optional[Path]:
        """Synthesise to a WAV file. Returns None instead of raising."""
        try:
            import soundfile as sf

            audio = self.speak_array(text, language)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, self.sample_rate)
            return Path(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"TTS failed for one line, continuing silently - {exc}")
            return None


class IndicTTS(_Engine):
    """Indic Parler-TTS. Loaded lazily; needs the `parler_tts` package."""

    name = "parler"

    def __init__(self, model_id: str = None, device: str = None):
        super().__init__()
        self.model_id = model_id or MODEL_ID
        self.device = device
        self._model = None
        self.sample_rate = 44100

    def _load(self) -> None:
        if self._model is not None or self._failed:
            return
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"loading TTS {self.model_id} on {self.device} ...")
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._desc_tok = AutoTokenizer.from_pretrained(self._model.config.text_encoder._name_or_path)
            self.sample_rate = self._model.config.sampling_rate
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._failed = f"Parler-TTS unavailable: {exc}"

    def synth(self, text: str, language: str):
        import torch

        # Same line, same voice: sampling is seeded from the text.
        torch.manual_seed(int(cache_key(text, language, self.name)[:8], 16))
        d = self._desc_tok(VOICE.get(language, VOICE["english"]), return_tensors="pt").to(self.device)
        p = self._tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio = self._model.generate(
                input_ids=d.input_ids, attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
            )
        return audio.cpu().numpy().squeeze()


class MMSTTS(_Engine):
    """Meta MMS voices, one small model per language."""

    name = "mms"

    def __init__(self, device: str = None):
        super().__init__()
        self.device = device
        self._models = {}
        self._loaded_any = False

    def _load(self) -> None:
        if self._loaded_any or self._failed:
            return
        try:
            import torch
            from transformers import VitsModel  # noqa: F401

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._get("english")
            self._loaded_any = True
        except Exception as exc:  # noqa: BLE001
            self._failed = f"MMS-TTS unavailable: {exc}"

    def _get(self, language: str):
        code = MMS_CODE.get(language, "eng")
        if code not in self._models:
            from transformers import AutoTokenizer, VitsModel

            name = f"facebook/mms-tts-{code}"
            print(f"loading TTS {name} on {self.device} ...")
            model = VitsModel.from_pretrained(name).to(self.device)
            self._models[code] = (model, AutoTokenizer.from_pretrained(name))
            self.sample_rate = model.config.sampling_rate
        return self._models[code]

    def synth(self, text: str, language: str):
        import torch

        model, tok = self._get(language)
        if getattr(tok, "is_uroman", False):
            import uroman  # romanises scripts the model was trained on in Latin form

            text = uroman.Uroman().romanize_string(text)
        inputs = tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(int(cache_key(text, language, self.name)[:8], 16))
        with torch.no_grad():
            wav = model(**inputs).waveform
        return wav.cpu().numpy().squeeze()


def local_engine() -> _Engine:
    """Parler if it loads, else MMS. Used inside the TTS server."""
    choice = os.environ.get("STORYTUTOR_TTS_ENGINE", "auto").lower()
    if choice in {"auto", "parler"}:
        parler = IndicTTS()
        if parler.available or choice == "parler":
            return parler
        print(f"{parler.failure} -- falling back to MMS voices")
    return MMSTTS()


# ---------------------------------------------------------------- client

class RemoteTTS:
    """Talks to scripts/tts_server.py. Same speak() contract as the engines."""

    def __init__(self, url: str = None, timeout: float = 120):
        self.url = (url or TTS_URL).rstrip("/")
        self.timeout = timeout
        self._failed: Optional[str] = None
        self.engine: Optional[str] = None

    @property
    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=3) as r:
                info = json.loads(r.read().decode("utf-8"))
            self.engine = info.get("engine")
            self._failed = None if info.get("ok") else info.get("error", "server not ready")
        except Exception as exc:  # noqa: BLE001
            self._failed = f"no voice server at {self.url} ({exc})"
        return self._failed is None

    @property
    def failure(self) -> Optional[str]:
        return self._failed

    def wav_bytes(self, text: str, language: str) -> bytes:
        body = json.dumps({"text": text, "language": language}).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/tts", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def speak(self, text: str, language: str, out_path: Path) -> Optional[Path]:
        if not (text or "").strip():
            return None
        try:
            data = self.wav_bytes(text, language)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(data)
            return Path(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"TTS failed for one line, continuing silently - {exc}")
            return None


def get_tts():
    """The voice for this process: the TTS server if it is up, else in-process."""
    remote = RemoteTTS()
    if remote.available:
        print(f"voice server OK at {remote.url} · {remote.engine}")
        return remote
    # Loading a voice model inside the app or the video job is opt-in: in the
    # main environment it would quietly pull models the server exists to hold.
    if os.environ.get("STORYTUTOR_TTS_LOCAL", "").lower() in {"1", "true", "yes"}:
        local = local_engine()
        if local.available:
            return local
    return remote   # unavailable; its failure names the server as the fix


def audio_duration(path: Path) -> Optional[float]:
    """Seconds of audio. Falls back to the standard-library wave reader: if
    soundfile is missing, returning None made every scene its fixed length,
    and the render then cut longer narration off mid-sentence."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        pass
    try:
        import wave

        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None

