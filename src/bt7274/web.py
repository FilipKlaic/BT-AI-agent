"""
The web interface: BT's optic in a browser tab, served from your own computer.

`uv run bt-web` starts a local server and opens the page. It is the same BT as the
terminal — same brain, tools and ~/.bt7274/memory.json — with a different face. The
page records the Pilot as 16 kHz WAV and posts it to /api/talk, which streams back
newline-delimited JSON events: what BT heard, each sentence with its audio, and timers
for the page to announce. Speech is transcribed locally with faster-whisper when the
`listen` extra is installed, otherwise by Gemini.

The server only answers its own page: requests from other sites open in the same
browser are refused, so a web page can't drive BT or replace the API key.
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import wave
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .brain import Brain, GeminiBrain, SetupError, create_brain
from .config import PROJECT_ROOT, Settings
from .effects import radio_filter
from .listen import SAMPLE_RATE, create_ears
from .memory import Memory
from .speech import SentenceSplitter, clean_for_speech
from .tools import (
    Toolbox,
    message_for_timer,
    open_app_tool,
    open_url_tool,
    remember_tool,
    system_status_tool,
    time_tool,
    timer_tool,
    weather_tool,
)

STATIC_DIR = Path(__file__).parent / "static"

MAX_AUDIO_BYTES = 4 * 1024 * 1024  # about two minutes of 16 kHz mono WAV
MAX_TEXT_CHARS = 2000
API_KEY_PATTERN = re.compile(r"[A-Za-z0-9._-]{20,200}")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class WebVoice:
    """BT's Piper voice, rendering sentences to audio for the browser to play."""

    def __init__(self, model_path: Path):
        from piper import PiperVoice

        self._piper = PiperVoice.load(model_path)
        self.sample_rate = self._piper.config.sample_rate
        self._lock = threading.Lock()

    def render(self, text: str, radio: bool = False) -> np.ndarray:
        with self._lock:
            chunks = [chunk.audio_float_array for chunk in self._piper.synthesize(text)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return radio_filter(audio, self.sample_rate) if radio else audio


def read_wav(data: bytes) -> np.ndarray:
    """16-bit mono WAV, as the page records it, to float samples at Whisper's rate."""
    try:
        with wave.open(io.BytesIO(data)) as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                raise HTTPException(400, "Expected 16-bit mono WAV.")
            rate, frames = wav.getframerate(), wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise HTTPException(400, "That recording isn't a WAV file.") from error
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    if rate != SAMPLE_RATE and samples.size:
        positions = np.arange(0, samples.size, rate / SAMPLE_RATE)
        samples = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    return samples


def save_api_key(env_path: Path, key: str):
    """Write GEMINI_API_KEY into .env, keeping every other line as it was."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    entry = f"GEMINI_API_KEY={key}"
    for index, existing in enumerate(lines):
        if re.match(r"\s*GEMINI_API_KEY\s*=", existing):
            lines[index] = entry
            break
    else:
        lines.append(entry)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_path.chmod(0o600)


def describe_error(error: Exception) -> str:
    if isinstance(error, HTTPException):  # e.g. a bad WAV, raised once the stream has started
        return str(error.detail)
    message = str(error)
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return "Neural link at capacity: the free tier allows a few requests a minute. Stand by."
    if "503" in message or "UNAVAILABLE" in message:
        return "Neural link unavailable. The model is overloaded; try again shortly, Pilot."
    if "API key not valid" in message or "API_KEY_INVALID" in message:
        return "Google rejected the API key. Set a new one in the menu, Pilot."
    return f"Protocol error: {message[:300]}"


def line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


class Station:
    """BT himself, shared by every request: one Pilot, one conversation at a time."""

    def __init__(
        self,
        settings: Settings,
        memory: Memory,
        voice: Any = None,
        ears: Any = None,
        http: httpx.Client | None = None,
        client: Any = None,
        env_path: Path = PROJECT_ROOT / ".env",
    ):
        self.settings = settings
        self.memory = memory
        self.voice = voice
        self.ears = ears
        self.env_path = env_path
        self._client = client
        self._events: list[dict[str, Any]] = []
        self._radio = False
        self._busy = threading.Lock()
        self.toolbox = Toolbox(
            [
                time_tool(),
                system_status_tool(),
                timer_tool(self._start_timer),
                weather_tool(http or httpx.Client(timeout=10)),
                open_url_tool(),
                open_app_tool(),
                remember_tool(memory),
            ]
        )
        self.brain: Brain | None = None if self.setup_needed else self._make_brain()

    @property
    def setup_needed(self) -> bool:
        return self.settings.backend == "gemini" and not self.settings.gemini_api_key

    def _make_brain(self) -> Brain:
        if self.settings.backend == "gemini":
            return GeminiBrain(
                self.settings.gemini_api_key or "",
                self.settings.model,
                self.memory,
                self.toolbox,
                self.settings.history_turns,
                client=self._client,
            )
        return create_brain(self.settings, self.memory, self.toolbox)

    def set_api_key(self, key: str, check: bool = True):
        """Check the key with Google (no generation quota used), save it and reconnect."""
        if check:
            from google import genai

            try:
                genai.Client(api_key=key).models.get(model=self.settings.model)
            except Exception as error:
                raise HTTPException(400, "Google rejected that key. Check it, Pilot.") from error
        save_api_key(self.env_path, key)
        self.settings.gemini_api_key = key
        os.environ["GEMINI_API_KEY"] = key
        self._client = None
        self.brain = self._make_brain()

    def _start_timer(self, seconds: int, label: str):
        # Timers are announced by the page, so they ring wherever the Pilot is looking.
        message = message_for_timer(label)
        self._events.append(
            {"type": "timer", "seconds": seconds, "label": label, "text": message}
            | self.audio(message, self._radio)
        )

    def audio(self, text: str, radio: bool) -> dict[str, Any]:
        """Audio for one sentence, as base64 16-bit PCM the browser turns into a buffer."""
        spoken = clean_for_speech(text)
        if self.voice is None or not spoken:
            return {}
        try:
            samples = self.voice.render(spoken, radio)
        except Exception:
            return {}  # the caption still shows; one silent sentence beats a failed reply
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return {"audio": base64.b64encode(pcm).decode("ascii"), "rate": self.voice.sample_rate}

    def hear(self, recording: bytes) -> str:
        if self.ears is not None:
            return self.ears.transcribe(read_wav(recording))
        if isinstance(self.brain, GeminiBrain):
            return self.brain.transcribe(recording)
        raise RuntimeError("Speech recognition needs faster-whisper: uv sync --extra listen")

    def converse(self, text: str, recording: bytes, radio: bool) -> Iterator[str]:
        if not self._busy.acquire(timeout=15):
            yield line({"type": "error", "message": "Still processing the last order, Pilot."})
            yield line({"type": "done"})
            return
        try:
            yield from self._converse(text, recording, radio)
        finally:
            self._busy.release()

    def _converse(self, text: str, recording: bytes, radio: bool) -> Iterator[str]:
        self._radio = radio
        self._events.clear()
        brain = self.brain
        if brain is None:
            yield line({"type": "error", "message": "Add a Gemini API key first, Pilot."})
            yield line({"type": "done"})
            return

        def flush_events() -> Iterator[str]:
            while self._events:
                yield line(self._events.pop(0))

        try:
            if recording:
                text = self.hear(recording)
                yield line({"type": "heard", "text": text})
                if not text:
                    yield line({"type": "error", "message": "No speech detected, Pilot."})
                    yield line({"type": "done"})
                    return
            splitter = SentenceSplitter()
            for chunk in brain.reply(text):
                for sentence in splitter.feed(chunk):
                    yield line({"type": "say", "text": sentence} | self.audio(sentence, radio))
                yield from flush_events()
            if rest := splitter.flush():
                yield line({"type": "say", "text": rest} | self.audio(rest, radio))
            yield from flush_events()
        except Exception as error:
            yield line({"type": "error", "message": describe_error(error)})
        yield line({"type": "done"})

    def forget(self):
        with self._busy:
            self.memory.forget_everything()
            if self.brain is not None:
                self.brain.forget()


def create_app(station: Station) -> FastAPI:
    app = FastAPI(title="BT-7274", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def only_our_own_page(request: Request, call_next):
        # Refuse other hostnames (DNS rebinding) and requests sent by other websites.
        host = request.headers.get("host", "")
        if urlsplit(f"//{host}").hostname not in LOCAL_HOSTS:
            return JSONResponse({"detail": "BT only answers on localhost."}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin not in (None, f"http://{host}"):
            return JSONResponse({"detail": "Requests from other sites are refused."}, 403)
        return await call_next(request)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "setup_needed": station.setup_needed,
            "brain": station.brain.label if station.brain else None,
            "voice": station.voice is not None,
            "ears": "whisper" if station.ears is not None else "gemini",
            "facts": len(station.memory.facts),
            "exchanges": len(station.memory.exchanges),
        }

    @app.post("/api/setup")
    def setup(api_key: Annotated[str, Form()]) -> dict[str, Any]:
        key = api_key.strip()
        if not API_KEY_PATTERN.fullmatch(key):
            raise HTTPException(400, "That doesn't look like a Gemini API key.")
        if station.settings.backend != "gemini":
            raise HTTPException(400, "BT is running on Ollama; no API key needed.")
        station.set_api_key(key)
        return {"ok": True, "brain": station.brain.label if station.brain else None}

    @app.get("/api/memory")
    def memory() -> dict[str, Any]:
        return {"facts": station.memory.facts, "exchanges": len(station.memory.exchanges)}

    @app.delete("/api/memory")
    def forget() -> dict[str, Any]:
        station.forget()
        return {"ok": True}

    @app.post("/api/talk")
    async def talk(
        text: Annotated[str, Form()] = "",
        audio: Annotated[UploadFile | None, File()] = None,
        radio: Annotated[bool, Form()] = False,
    ) -> StreamingResponse:
        recording = await audio.read(MAX_AUDIO_BYTES + 1) if audio else b""
        if len(recording) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Recording too long, Pilot.")
        text = text.strip()[:MAX_TEXT_CHARS]
        if not text and not recording:
            raise HTTPException(400, "Nothing to send.")
        return StreamingResponse(
            station.converse(text, recording, radio),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="bt-web", description="BT-7274 in your browser.")
    parser.add_argument("--port", type=int, default=int(os.getenv("BT_WEB_PORT", "7274")))
    parser.add_argument("--no-browser", action="store_true", help="Don't open the page.")
    parser.add_argument("--no-memory", action="store_true", help="Don't load or save pilot data.")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    memory = Memory.load(None if args.no_memory else settings.data_dir / "memory.json")
    try:
        voice = WebVoice(settings.voice_model)
    except Exception as error:
        voice = None
        print(f"Voice system offline, replies will be text only: {error}")
    ears, ears_status = create_ears(settings.whisper_model)
    print(f"Audio receptors: {ears_status if ears else 'Gemini (add --extra listen for local)'}")
    try:
        station = Station(settings, memory, voice=voice, ears=ears)
    except SetupError as error:
        print(error)
        return 1

    url = f"http://127.0.0.1:{args.port}"
    print(f"BT-7274 online at {url}  (Ctrl+C to power down)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(station), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
