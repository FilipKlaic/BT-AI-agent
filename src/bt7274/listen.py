"""
BT's audio receptors: push-to-talk speech recognition with faster-whisper.

Optional: install with `uv sync --extra listen`. The Whisper model is downloaded
on first use (about 150 MB for base.en) and loaded in the background at startup.
"""

import os
import threading
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16_000  # what Whisper expects
MIN_SECONDS = 0.3


class Ears:
    def __init__(self, model_name: str = "base.en"):
        # The first-run model download would otherwise draw progress bars over the prompt.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from faster_whisper import WhisperModel  # ImportError means the extra isn't installed

        self.model_name = model_name
        self._model: WhisperModel | None = None
        self._error: Exception | None = None
        self._loaded = threading.Event()

        def load():
            try:
                self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            except Exception as error:
                self._error = error
            finally:
                self._loaded.set()

        threading.Thread(target=load, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self._loaded.is_set()

    def record(self, until: Callable[[], object]) -> np.ndarray:
        """Record from the microphone until `until()` returns (e.g. the Pilot presses Enter)."""
        import sounddevice as sd

        frames: list[np.ndarray] = []
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=lambda data, _frames, _time, _status: frames.append(data.copy()),
        ):
            until()
        return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE * MIN_SECONDS:
            return ""
        self._loaded.wait()
        if self._model is None:
            raise RuntimeError(f"Speech model failed to load: {self._error}")
        segments, _info = self._model.transcribe(
            audio,
            language="en" if self.model_name.endswith(".en") else None,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def create_ears(model_name: str) -> tuple[Ears | None, str]:
    try:
        return Ears(model_name), f"Push-to-talk ({model_name})"
    except ImportError:
        return None, "Not installed (uv sync --extra listen)"
