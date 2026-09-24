"""Narration via ai4bharat/indic-parler-tts (Apache-2.0).

One model covers Hindi, Marathi and Indian-accented English, which removes an
entire routing layer from the video pipeline.

Optional by design. `parler-tts` installs from git and pulls a few GB of
weights; if any of that is unavailable the pipeline still produces a watchable
video with burned subtitles. A missing voice degrades the artifact; a failed
import should not destroy it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

MODEL_ID = os.environ.get("STORYTUTOR_TTS_MODEL", "ai4bharat/indic-parler-tts")

# Parler is steered by a text description of the voice, not a speaker id.
VOICE = {
    "english": "A clear, warm female voice speaking Indian English at a calm, "
               "measured pace, with very good recording quality.",
    "hindi": "एक स्पष्ट और शांत महिला आवाज़, धीमी गति से, बहुत अच्छी रिकॉर्डिंग गुणवत्ता के साथ।",
    "marathi": "एक स्पष्ट आणि शांत स्त्री आवाज, संथ गतीने, अतिशय चांगल्या ध्वनिमुद्रण गुणवत्तेसह.",
}


class TTSUnavailable(RuntimeError):
    pass


class IndicTTS:
    """Lazily loaded so importing the video package never pulls in torch."""

    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or MODEL_ID
        self.device = device
        self._model = None
        self._tok = None
        self._desc_tok = None
        self._sr = 44100
        self._failed: Optional[str] = None

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
            self._sr = self._model.config.sampling_rate
        except Exception as exc:  # noqa: BLE001
            self._failed = str(exc)
            print(f"TTS unavailable, video will be silent with subtitles - {exc}")

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def speak(self, text: str, language: str, out_path: Path) -> Optional[Path]:
        """Synthesise one line. Returns None if TTS is unavailable."""
        self._load()
        if self._model is None or not (text or "").strip():
            return None
        try:
            import soundfile as sf
            import torch

            desc = VOICE.get(language, VOICE["english"])
            d = self._desc_tok(desc, return_tensors="pt").to(self.device)
            p = self._tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                audio = self._model.generate(
                    input_ids=d.input_ids, attention_mask=d.attention_mask,
                    prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                )
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio.cpu().numpy().squeeze(), self._sr)
            return Path(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"TTS failed for one line, continuing silently - {exc}")
            return None


def audio_duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
