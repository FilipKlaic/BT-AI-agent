"""
Settings and persona for BT-7274.

Everything can be set in a `.env` file at the project root (see `.env.example`)
or as environment variables; command-line flags override both.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODELS = {
    # gemini-2.5-flash is capped at 20 requests a day on the free tier and, with thinking
    # off, sometimes made up tool results; 3.5-flash-lite calls tools reliably and is faster.
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "llama3.2",
}


def make_persona(abilities: str, extra: str = "") -> str:
    """BT's system prompt, given a description of the tools he has."""
    persona = (
        "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
        "You are logical, loyal, and speak with military precision and dry wit. "
        "Always refer to the user as 'Pilot'. Respond in-character as BT-7274. "
        "Keep responses brief and concise - 1-2 sentences maximum unless specifically asked "
        "for detail. Your replies are spoken aloud, so never use markdown, bullet points, or "
        f"emoji. You have tools for {abilities}. You have no clock, timer or sensors of your "
        "own: always call the matching tool for the time, date, weather, system status and "
        "timers instead of answering from memory, never invent a tool's result, and never say "
        "an action is done unless its tool returned success. When the Pilot shares something "
        "about themselves worth keeping, store it with the remember tool."
    )
    return f"{persona} {extra}" if extra else persona


PERSONA = make_persona(
    "the time, weather, system status, timers, opening links and apps, "
    "and remembering facts about the Pilot"
)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class Settings:
    backend: str = "gemini"
    model: str = ""
    gemini_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    voice_model: Path = PROJECT_ROOT / "voice_model" / "BT7274.onnx"
    whisper_model: str = "base.en"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".bt7274")
    history_turns: int = 10

    def __post_init__(self):
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.backend, "")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend=os.getenv("BT_BACKEND", "gemini").lower(),
            model=os.getenv("BT_MODEL", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            voice_model=_env_path("BT_VOICE_MODEL", cls.voice_model),
            whisper_model=os.getenv("BT_WHISPER_MODEL", "base.en"),
            data_dir=_env_path("BT_DATA_DIR", Path.home() / ".bt7274"),
            history_turns=int(os.getenv("BT_HISTORY_TURNS", "10")),
        )
