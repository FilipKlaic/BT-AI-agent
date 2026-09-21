"""
The terminal interface: boot sequence, chat loop and commands.
"""

import time

from rich.console import Console
from rich.table import Table

from .brain import Brain
from .listen import Ears
from .memory import Memory
from .speech import SentenceSplitter
from .voice import PiperVoice, Voice

try:
    import readline  # noqa: F401  (arrow keys and history for input())
except ImportError:
    pass

PILOT = "[bold cyan]Pilot:[/] "
BT = "[bold orange3]BT-7274:[/] "

EXIT_WORDS = {"exit", "quit", "shutdown"}
MODES = {"text+voice": "Text + Voice", "text": "Text Only", "voice": "Voice Only"}

COMMANDS = [
    ("Enter (empty line)", "Interrupt BT / push-to-talk: speak, then press Enter again"),
    ("/talk", "Push-to-talk"),
    ("/hush", "Stop BT talking"),
    ("/voice", "Cycle Text + Voice / Text Only / Voice Only"),
    ("/radio", "Toggle the comms radio effect on BT's voice"),
    ("/memory", "Show what BT remembers about you"),
    ("/forget", "Wipe all pilot data"),
    ("/help", "Show this list"),
    ("exit, quit, shutdown", "Power down BT"),
]


class Terminal:
    def __init__(
        self,
        console: Console,
        brain: Brain,
        memory: Memory,
        voice: Voice | None,
        voice_status: str,
        ears: Ears | None,
        ears_status: str,
        animate: bool = True,
        mode: str = "text+voice",
    ):
        self.console = console
        self.brain = brain
        self.memory = memory
        self.voice = voice
        self.voice_status = voice_status
        self.ears = ears
        self.ears_status = ears_status
        self.animate = animate
        self.mode = mode if voice else "text"

    # --- main loop ---------------------------------------------------------------

    def run(self):
        self.boot()
        try:
            while True:
                try:
                    line = self.console.input(PILOT).strip()
                except (KeyboardInterrupt, EOFError):
                    self.console.print()
                    break
                if line.lower().lstrip("/") in EXIT_WORDS:
                    break
                if not line:
                    self.interrupt_or_listen()
                elif line.startswith("/"):
                    self.command(line)
                else:
                    self.respond(line)
        finally:
            self.shutdown()

    def respond(self, text: str):
        speak = self.voice is not None and self.mode != "text"
        show_text = self.mode != "voice"
        if self.voice:
            self.voice.stop()  # a new message cuts off whatever BT was still saying

        splitter = SentenceSplitter()
        self.console.print(BT, end="")
        if not show_text:
            self.console.print("[dim](transmitting audio)[/]", end="")
        received = False
        try:
            for chunk in self.brain.reply(text):
                received = True
                if show_text:
                    self.console.out(chunk, end="", highlight=False)
                if speak:
                    for sentence in splitter.feed(chunk):
                        self.voice.say(sentence)
            if speak:
                self.voice.say(splitter.flush())
            if not received:
                self.console.print("[dim]No response received. Please retry, Pilot.[/]", end="")
        except KeyboardInterrupt:
            self._hush()
            self.console.print(" [dim](interrupted)[/]", end="")
        except Exception as error:
            self._hush()
            self.console.print(f"\n[red]Protocol error:[/] {error}", end="")
        self.console.print("\n")

    def alert(self, message: str):
        """Called from a timer thread: announce, then redraw the prompt."""
        self.console.print(f"\n{BT}{message}")
        if self.voice and self.mode != "text":
            self.voice.say(message)
        self.console.print(PILOT, end="")

    # --- voice and listening -----------------------------------------------------

    def _hush(self):
        if self.voice:
            self.voice.stop()

    def interrupt_or_listen(self):
        was_speaking = self.voice is not None and self.voice.speaking
        self._hush()
        if self.ears:
            self.talk()
        elif not was_speaking:
            self.console.print("[dim]Type a message, or /help for commands.[/]")

    def talk(self):
        if not self.ears:
            self.console.print(f"[yellow]Audio receptors offline:[/] {self.ears_status}")
            return
        self._hush()
        self.console.print("[bold green]● Listening[/] [dim]- press Enter to send[/] ", end="")
        try:
            audio = self.ears.record(until=input)
            status = "Processing audio..."
            if not self.ears.ready:
                status = "Loading speech model (the first run downloads it, ~150 MB)..."
            with self.console.status(status):
                text = self.ears.transcribe(audio)
        except KeyboardInterrupt:
            self.console.print("\n[dim]Cancelled.[/]")
            return
        except Exception as error:
            self.console.print(f"[red]Audio receptor error:[/] {error}")
            return
        if not text:
            self.console.print("[dim]No speech detected.[/]")
            return
        self.console.print(f"{PILOT}{text}", highlight=False)
        self.respond(text)

    # --- commands ----------------------------------------------------------------

    def command(self, line: str):
        name = line[1:].split(maxsplit=1)[0].lower() if len(line) > 1 else ""
        handlers = {
            "help": self.show_help,
            "talk": self.talk,
            "hush": self._hush,
            "voice": self.cycle_mode,
            "radio": self.toggle_radio,
            "memory": self.show_memory,
            "forget": self.forget,
        }
        handler = handlers.get(name)
        if handler is None:
            self.console.print(f"[yellow]Unknown command '{line}'.[/] Type /help for commands.")
        else:
            handler()

    def show_help(self):
        table = Table(show_header=False, box=None, padding=(0, 2))
        for keys, description in COMMANDS:
            table.add_row(f"[bold]{keys}[/]", description)
        self.console.print(table)

    def cycle_mode(self):
        if not self.voice:
            self.console.print("[yellow]Voice system offline.[/]")
            return
        modes = list(MODES)
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        if self.mode == "text":
            self._hush()
        self.console.print(f"Voice mode: [bold]{MODES[self.mode]}[/]")

    def toggle_radio(self):
        if not isinstance(self.voice, PiperVoice):
            self.console.print("[yellow]The radio effect needs BT's Piper voice.[/]")
            return
        self.voice.radio = not self.voice.radio
        self.console.print(f"Comms radio effect: [bold]{'On' if self.voice.radio else 'Off'}[/]")

    def show_memory(self):
        if self.memory.facts:
            self.console.print("[bold]What BT knows about you:[/]")
            for fact in self.memory.facts:
                self.console.print(f"  • {fact}", highlight=False)
        else:
            self.console.print("BT hasn't stored any facts about you yet.")
        where = self.memory.path or "not saved (--no-memory)"
        self.console.print(f"[dim]{len(self.memory.exchanges)} past exchanges · {where}[/]")

    def forget(self):
        answer = self.console.input("Wipe all pilot data? This can't be undone. [y/N] ")
        if answer.strip().lower() == "y":
            self.memory.forget_everything()
            self.brain.forget()
            self.console.print("Pilot data erased.")

    # --- boot and shutdown -------------------------------------------------------

    def _pause(self, seconds: float):
        if self.animate:
            time.sleep(seconds)

    def boot(self):
        facts, exchanges = len(self.memory.facts), len(self.memory.exchanges)
        pilot_data = (
            f"{facts} facts, {exchanges} exchanges on file" if facts or exchanges else "None"
        )
        if self.memory.path is None:
            pilot_data = "Not saved (--no-memory)"
        statuses = [
            ("Chassis", "Operational"),
            ("Neural link", self.brain.label),
            ("Voice system", self.voice_status),
            ("Audio receptors", self.ears_status),
            ("Pilot data", pilot_data),
        ]

        self.console.rule("[bold]BT-7274 VANGUARD-CLASS TITAN ONLINE[/]")
        for label, status in statuses:
            color = "green"
            if status.startswith(("Fallback", "Not installed", "None")):
                color = "yellow"
            elif status.startswith(("Offline", "Disabled", "Not saved")):
                color = "red"
            self.console.print(f"  {label:<16} [{color}]{status}[/]", highlight=False)
            self._pause(0.08)
        self.console.rule()

        greeting = "Welcome back, Pilot." if exchanges else "Hello, Pilot. I am BT-7274."
        greeting += " Ready for orders."
        self.console.print(f"{BT}{greeting}")
        if self.voice and self.mode != "text":
            self.voice.say(greeting)
        hint = "Type to talk, /help for commands."
        if self.ears:
            hint = "Type to talk, or press Enter on an empty line to speak. /help for commands."
        self.console.print(f"[dim]{hint}[/]\n")

    def shutdown(self):
        self.memory.save()
        farewell = "Until next time, Pilot."
        if self.voice:
            self.voice.stop()
            if self.mode != "text":
                self.voice.say(farewell)
        saved = "Complete." if self.memory.path else "Skipped (--no-memory)."
        for line in [
            "Initiating shutdown sequence...",
            f"Saving pilot data... {saved}",
            "Neural link terminated.",
            farewell,
            "BT-7274 systems offline.",
        ]:
            self.console.print(f"[dim]{line}[/]")
            self._pause(0.15)
        if self.voice:
            self.voice.wait(timeout=5)
            self.voice.close()
