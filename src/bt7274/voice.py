"""
BT-7274's voice.

`PiperVoice` loads the BT voice model once and runs a two-stage pipeline: one
thread synthesizes sentences while another plays them, so BT starts talking as
soon as the first sentence of a reply is ready. `SayVoice` is a fallback that
uses the macOS `say` command when the Piper model can't be loaded.
"""

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from .effects import radio_filter
from .speech import clean_for_speech

# Frames written to the audio device per call; small enough that stop() is near-instant.
_PLAYBACK_BLOCK = 1024


class _Pipeline:
    """Tracks queued speech so callers can interrupt it or wait for it to finish."""

    def __init__(self):
        self._epoch = 0  # bumped by stop(); anything queued in an older epoch is dropped
        self._pending = 0
        self._idle = threading.Condition()

    def _begin(self) -> int:
        with self._idle:
            self._pending += 1
            return self._epoch

    def _finish(self):
        with self._idle:
            self._pending -= 1
            if self._pending == 0:
                self._idle.notify_all()

    def _is_stale(self, epoch: int) -> bool:
        return epoch != self._epoch

    @property
    def speaking(self) -> bool:
        return self._pending > 0

    def stop(self):
        """Drop everything queued and cut off the sentence being spoken."""
        with self._idle:
            self._epoch += 1

    def wait(self, timeout: float | None = None) -> bool:
        """Block until BT has finished talking. Returns False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)


class PiperVoice(_Pipeline):
    name = "Piper (BT-7274 model)"

    def __init__(self, model_path: Path, radio: bool = False):
        super().__init__()
        import sounddevice as sd
        from piper import PiperVoice as _Piper

        self._piper = _Piper.load(model_path)
        self.sample_rate = self._piper.config.sample_rate
        self.radio = radio

        self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32")
        self._stream.start()

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._synthesize_loop, daemon=True),
            threading.Thread(target=self._playback_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def say(self, text: str):
        """Queue a sentence to be spoken. Returns immediately."""
        text = clean_for_speech(text)
        if text:
            self._text_queue.put((self._begin(), text))

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [chunk.audio_float_array for chunk in self._piper.synthesize(text)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return radio_filter(audio, self.sample_rate) if self.radio else audio

    def _synthesize_loop(self):
        while (item := self._text_queue.get()) is not None:
            epoch, text = item
            if self._is_stale(epoch):
                self._finish()
                continue
            try:
                audio = self.synthesize(text)
            except Exception:
                audio = np.zeros(0, dtype=np.float32)
            self._audio_queue.put((epoch, audio))
        self._audio_queue.put(None)

    def _playback_loop(self):
        while (item := self._audio_queue.get()) is not None:
            epoch, audio = item
            for start in range(0, audio.size, _PLAYBACK_BLOCK):
                if self._is_stale(epoch):
                    break
                self._stream.write(np.ascontiguousarray(audio[start : start + _PLAYBACK_BLOCK]))
            self._finish()

    def close(self):
        self.stop()
        self._text_queue.put(None)
        for thread in self._threads:
            thread.join(timeout=2)
        self._stream.stop()
        self._stream.close()


class SayVoice(_Pipeline):
    """Fallback voice using the macOS `say` command. Not BT's real voice, but close enough."""

    name = "macOS say (fallback)"
    radio = False

    def __init__(self):
        super().__init__()
        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._thread.start()

    def say(self, text: str):
        text = clean_for_speech(text)
        if text:
            self._queue.put((self._begin(), text))

    def _speak_loop(self):
        while (item := self._queue.get()) is not None:
            epoch, text = item
            if not self._is_stale(epoch):
                # Arguments are passed as a list (no shell), so any text is safe here.
                self._process = subprocess.Popen(["say", "-r", "170", text])
                self._process.wait()
                self._process = None
            self._finish()

    def stop(self):
        super().stop()
        if self._process is not None:
            self._process.terminate()

    def close(self):
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=2)


Voice = PiperVoice | SayVoice


def create_voice(model_path: Path, radio: bool = False) -> tuple[Voice | None, str]:
    """Return (voice, status message). Falls back to `say`, then to no voice at all."""
    try:
        return PiperVoice(model_path, radio=radio), "Online"
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    if shutil.which("say"):
        return SayVoice(), f"Fallback ({reason})"
    return None, f"Offline ({reason})"
