import io

import numpy as np
import pytest
from rich.console import Console

from bt7274 import voice as voice_module
from bt7274.config import Settings
from bt7274.memory import Memory
from bt7274.terminal import Terminal


class FakeStream:
    """Replaces the sound card: records what would have been played."""

    played: list[np.ndarray] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def write(self, block):
        FakeStream.played.append(block.copy())

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def piper_voice(monkeypatch):
    model = Settings().voice_model
    if not model.exists():
        pytest.skip("voice model not present")
    import sounddevice

    FakeStream.played = []
    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
    voice = voice_module.PiperVoice(model)
    yield voice
    voice.close()


def test_piper_voice_speaks_queued_sentences(piper_voice):
    piper_voice.say("Voice systems operational, Pilot.")
    piper_voice.say("Protocol three: protect the Pilot.")
    assert piper_voice.wait(timeout=30)
    seconds = sum(block.size for block in FakeStream.played) / piper_voice.sample_rate
    assert 2.0 < seconds < 10.0
    assert not piper_voice.speaking


def test_piper_voice_stop_drops_queued_speech(piper_voice):
    for _ in range(5):
        piper_voice.say("This is a long sentence that BT should never finish saying out loud.")
    piper_voice.stop()
    assert piper_voice.wait(timeout=30)
    seconds = sum(block.size for block in FakeStream.played) / piper_voice.sample_rate
    assert seconds < 5.0  # far less than the ~20 seconds that were queued


def test_radio_effect_is_applied_when_enabled(piper_voice):
    # Piper normalizes to a peak of 1.0; the radio filter scales to 0.9.
    assert np.max(np.abs(piper_voice.synthesize("Pilot."))) > 0.99
    piper_voice.radio = True
    radio = piper_voice.synthesize("Pilot.")
    assert radio.size > 0 and np.max(np.abs(radio)) <= 0.9 + 1e-6


# --- terminal ------------------------------------------------------------------------


class FakeBrain:
    label = "fake"

    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.forgotten = False

    def reply(self, text):
        yield from self.chunks
        if self.error:
            raise self.error

    def forget(self):
        self.forgotten = True


class FakeVoice:
    def __init__(self):
        self.said, self.stops, self.speaking = [], 0, False

    def say(self, text):
        if text:
            self.said.append(text)

    def stop(self):
        self.stops += 1


def make_terminal(brain, voice=None):
    output = io.StringIO()
    console = Console(file=output, width=100)
    terminal = Terminal(
        console, brain, Memory.load(None), voice, "Online", None, "Disabled", animate=False
    )
    return terminal, output


def test_reply_is_printed_and_spoken_sentence_by_sentence():
    voice = FakeVoice()
    terminal, output = make_terminal(FakeBrain(["Affirmative. Pro", "tocol three engaged."]), voice)
    terminal.respond("Go")
    assert voice.said == ["Affirmative.", "Protocol three engaged."]
    assert "Affirmative. Protocol three engaged." in output.getvalue()
    assert voice.stops == 1  # the previous reply is cut off before the new one starts


def test_text_only_mode_does_not_speak():
    voice = FakeVoice()
    terminal, _ = make_terminal(FakeBrain(["Silent running."]), voice)
    terminal.cycle_mode()  # text+voice -> text
    terminal.respond("Go")
    assert voice.said == []


def test_backend_errors_are_reported_not_raised():
    terminal, output = make_terminal(FakeBrain(["Par"], error=RuntimeError("quota exceeded")))
    terminal.respond("Go")
    assert "Protocol error" in output.getvalue() and "quota exceeded" in output.getvalue()


def test_unknown_commands_do_not_crash():
    terminal, output = make_terminal(FakeBrain())
    terminal.command("/tts")  # used to crash the old app with an AttributeError
    assert "Unknown command" in output.getvalue()


def test_forget_wipes_memory(monkeypatch):
    brain = FakeBrain()
    terminal, _ = make_terminal(brain)
    terminal.memory.add_fact("Callsign: Cooper")
    monkeypatch.setattr(terminal.console, "input", lambda prompt: "y")
    terminal.forget()
    assert terminal.memory.facts == [] and brain.forgotten
