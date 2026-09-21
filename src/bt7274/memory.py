"""
Pilot data: facts BT has been asked to remember plus recent conversation,
saved as JSON so BT picks up where you left off.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# How many past exchanges to keep on disk. The brain only sends the most recent few.
MAX_SAVED_EXCHANGES = 50


@dataclass
class Exchange:
    pilot: str
    bt: str


@dataclass
class Memory:
    path: Path | None = None  # None keeps everything in RAM (--no-memory)
    facts: list[str] = field(default_factory=list)
    exchanges: list[Exchange] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None) -> "Memory":
        memory = cls(path=path)
        if path is None or not path.exists():
            return memory
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A damaged file shouldn't stop BT from booting; keep a copy and start fresh.
            path.replace(path.with_suffix(".corrupt.json"))
            return memory
        memory.facts = [str(fact) for fact in data.get("facts", [])]
        memory.exchanges = [Exchange(**item) for item in data.get("exchanges", [])]
        return memory

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "facts": self.facts,
            "exchanges": [asdict(item) for item in self.exchanges[-MAX_SAVED_EXCHANGES:]],
        }
        # Write to a temp file and swap it in, so a crash mid-write can't corrupt the memory.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add_exchange(self, pilot: str, bt: str):
        self.exchanges.append(Exchange(pilot, bt))
        self.save()

    def add_fact(self, fact: str) -> bool:
        fact = fact.strip()
        if not fact or fact.lower() in (known.lower() for known in self.facts):
            return False
        self.facts.append(fact)
        self.save()
        return True

    def forget_everything(self):
        self.facts.clear()
        self.exchanges.clear()
        self.save()
