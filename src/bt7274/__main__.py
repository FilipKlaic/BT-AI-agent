"""
Command-line entry point: run `uv run bt` (or `python -m bt7274`).
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console

from .brain import SetupError, create_brain
from .config import DEFAULT_MODELS, Settings
from .listen import create_ears
from .memory import Memory
from .terminal import Terminal
from .tools import build_toolbox
from .voice import create_voice


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bt", description="Talk to BT-7274 in your terminal.")
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), help="Language model backend.")
    parser.add_argument("--model", help="Model name, e.g. gemini-3.5-flash-lite or llama3.2.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text-only", action="store_true", help="Don't load BT's voice.")
    mode.add_argument("--voice-only", action="store_true", help="Speak replies without printing.")
    parser.add_argument("--radio", action="store_true", help="Comms radio effect on the voice.")
    parser.add_argument("--no-listen", action="store_true", help="Don't load speech recognition.")
    parser.add_argument("--no-memory", action="store_true", help="Don't load or save pilot data.")
    parser.add_argument("--fast", action="store_true", help="Skip the boot/shutdown animation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    if args.backend and args.backend != settings.backend:
        settings.backend = args.backend
        settings.model = args.model or DEFAULT_MODELS[args.backend]
    elif args.model:
        settings.model = args.model

    console = Console(highlight=False)  # no auto-colouring of numbers and paths in BT's text
    memory = Memory.load(None if args.no_memory else settings.data_dir / "memory.json")
    terminal: Terminal | None = None

    def on_alert(message: str):
        if terminal is not None:
            terminal.alert(message)

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Loading the voice model takes a second or two; do it while the rest starts up.
        voice_job = (
            None if args.text_only else pool.submit(create_voice, settings.voice_model, args.radio)
        )
        ears, ears_status = (
            (None, "Disabled (--no-listen)")
            if args.no_listen
            else create_ears(settings.whisper_model)
        )
        try:
            brain = create_brain(settings, memory, build_toolbox(memory, on_alert))
        except SetupError as error:
            console.print(f"[red]{error}[/]")
            if voice_job and (voice := voice_job.result()[0]):
                voice.close()
            return 1
        voice, voice_status = voice_job.result() if voice_job else (None, "Disabled (--text-only)")

    terminal = Terminal(
        console,
        brain,
        memory,
        voice,
        voice_status,
        ears,
        ears_status,
        animate=not args.fast,
        mode="voice" if args.voice_only else "text+voice",
    )
    terminal.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
