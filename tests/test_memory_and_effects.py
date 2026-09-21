import json

import numpy as np

from bt7274.effects import radio_filter
from bt7274.memory import MAX_SAVED_EXCHANGES, Memory


def test_memory_round_trip(tmp_path):
    path = tmp_path / "data" / "memory.json"
    memory = Memory.load(path)
    memory.add_fact("Pilot's name is Filip")
    memory.add_exchange("Hello", "Hello, Pilot.")

    reloaded = Memory.load(path)
    assert reloaded.facts == ["Pilot's name is Filip"]
    assert reloaded.exchanges[0].bt == "Hello, Pilot."


def test_duplicate_facts_are_ignored(tmp_path):
    memory = Memory.load(tmp_path / "memory.json")
    assert memory.add_fact("Likes tea")
    assert not memory.add_fact("likes TEA ")
    assert not memory.add_fact("   ")
    assert memory.facts == ["Likes tea"]


def test_only_recent_exchanges_are_saved(tmp_path):
    path = tmp_path / "memory.json"
    memory = Memory.load(path)
    total = MAX_SAVED_EXCHANGES + 5
    for i in range(total):
        memory.add_exchange(f"q{i}", f"a{i}")
    saved = json.loads(path.read_text())
    assert len(saved["exchanges"]) == MAX_SAVED_EXCHANGES
    assert saved["exchanges"][-1] == {"pilot": f"q{total - 1}", "bt": f"a{total - 1}"}


def test_corrupt_memory_is_set_aside(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    memory = Memory.load(path)
    assert memory.facts == [] and memory.exchanges == []
    assert (tmp_path / "memory.corrupt.json").exists()


def test_memory_without_path_never_touches_disk(tmp_path):
    memory = Memory.load(None)
    memory.add_fact("secret")
    memory.forget_everything()
    assert list(tmp_path.iterdir()) == []


def test_radio_filter_keeps_length_and_level():
    sample_rate = 22050
    t = np.arange(sample_rate) / sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)).astype(
        np.float32
    )
    out = radio_filter(audio, sample_rate)
    assert out.shape == audio.shape and out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 0.9 + 1e-6
    assert radio_filter(np.zeros(0, dtype=np.float32), sample_rate).size == 0
