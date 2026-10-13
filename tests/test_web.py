import base64
import io
import json
import os
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import types
from test_brain import FakeGemini, chunk, text

from bt7274 import web
from bt7274.config import Settings
from bt7274.memory import Exchange, Memory
from bt7274.web import Station, create_app, read_wav

GOOD_KEY = "AIzaSyA-fake_key_for_tests_0123456789"


class FakeVoice:
    sample_rate = 22050

    def __init__(self):
        self.calls = []

    def render(self, text, radio=False):
        self.calls.append((text, radio))
        return np.full(100, 0.5, dtype=np.float32)


class FakeEars:
    def __init__(self, transcript):
        self.transcript = transcript
        self.heard = []

    def transcribe(self, samples):
        self.heard.append(samples)
        return self.transcript


class TranscribingGemini(FakeGemini):
    """FakeGemini that can also transcribe, like GeminiBrain.transcribe expects."""

    def __init__(self, *streams, transcript=""):
        super().__init__(*streams)
        self.transcript = transcript
        self.transcriptions = []

    def generate_content(self, model, contents, config):
        self.transcriptions.append(contents)
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model", parts=[types.Part.from_text(text=self.transcript)]
                    )
                )
            ]
        )


def wav_bytes(samples, rate=16000, channels=1, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def make_station(tmp_path, client=None, key="test-key", **kwargs):
    settings = Settings(gemini_api_key=key, data_dir=tmp_path)
    kwargs.setdefault("memory", Memory.load(None))
    return Station(settings, client=client, env_path=tmp_path / ".env", **kwargs)


def make_client(station):
    return TestClient(create_app(station), base_url="http://127.0.0.1:7274")


def events(response):
    return [json.loads(row) for row in response.text.splitlines()]


def kinds(response):
    return [event["type"] for event in events(response)]


# --- pages and middleware ------------------------------------------------------------


def test_serves_page_and_status(tmp_path):
    memory = Memory.load(None)
    memory.facts = ["Pilot likes tea"]
    memory.exchanges = [Exchange("a", "b")]
    station = make_station(tmp_path, FakeGemini(), memory=memory, voice=FakeVoice())
    client = make_client(station)

    assert client.get("/").status_code == 200
    assert client.get("/api/status").json() == {
        "setup_needed": False,
        "brain": "Gemini " + station.settings.model.removeprefix("models/"),
        "voice": True,
        "ears": "gemini",
        "facts": 1,
        "exchanges": 1,
    }


def test_status_reports_local_ears_and_missing_voice(tmp_path):
    station = make_station(tmp_path, FakeGemini(), ears=FakeEars("hi"))
    status = make_client(station).get("/api/status").json()
    assert (status["ears"], status["voice"]) == ("whisper", False)


def test_foreign_host_is_refused(tmp_path):
    app = create_app(make_station(tmp_path, FakeGemini()))
    assert TestClient(app).get("/api/status").status_code == 403  # "testserver"
    rebound = TestClient(app, base_url="http://evil.example:7274")
    assert rebound.get("/api/status").status_code == 403


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_local_hosts_are_allowed(tmp_path, host):
    app = create_app(make_station(tmp_path, FakeGemini()))
    assert TestClient(app, base_url=f"http://{host}:7274").get("/api/status").status_code == 200


def test_foreign_origin_cannot_post(tmp_path):
    station = make_station(tmp_path, FakeGemini([chunk(text("Online."))]))
    client = make_client(station)

    evil = client.post("/api/talk", data={"text": "hi"}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert (
        client.delete("/api/memory", headers={"Origin": "https://evil.example"}).status_code == 403
    )

    same = client.post(
        "/api/talk", data={"text": "hi"}, headers={"Origin": "http://127.0.0.1:7274"}
    )
    assert same.status_code == 200
    assert client.post("/api/talk", data={"text": "hi"}).status_code != 403


def test_foreign_origin_may_still_read(tmp_path):
    client = make_client(make_station(tmp_path, FakeGemini()))
    response = client.get("/api/status", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200


# --- talking -------------------------------------------------------------------------


def test_typed_text_streams_sentences_with_audio(tmp_path):
    gemini = FakeGemini([chunk(text("Affirmative, Pilot. ")), chunk(text("Standing by."))])
    voice = FakeVoice()
    station = make_station(tmp_path, gemini, voice=voice)

    response = make_client(station).post("/api/talk", data={"text": " Status? ", "radio": "true"})

    got = events(response)
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [e["type"] for e in got] == ["say", "say", "done"]
    assert [e["text"] for e in got[:2]] == ["Affirmative, Pilot.", "Standing by."]
    assert all(e["rate"] == 22050 for e in got[:2])
    pcm = np.frombuffer(base64.b64decode(got[0]["audio"]), dtype="<i2")
    assert pcm.size == 100 and pcm[0] == int(0.5 * 32767)
    assert voice.calls == [("Affirmative, Pilot.", True), ("Standing by.", True)]
    assert station.memory.exchanges == [Exchange("Status?", "Affirmative, Pilot. Standing by.")]


def test_reply_without_voice_is_caption_only(tmp_path):
    station = make_station(tmp_path, FakeGemini([chunk(text("Online."))]))
    got = events(make_client(station).post("/api/talk", data={"text": "hi"}))
    assert got == [{"type": "say", "text": "Online."}, {"type": "done"}]


def test_voice_failure_keeps_the_caption(tmp_path):
    class BrokenVoice(FakeVoice):
        def render(self, text, radio=False):
            raise RuntimeError("piper died")

    station = make_station(tmp_path, FakeGemini([chunk(text("Online."))]), voice=BrokenVoice())
    got = events(make_client(station).post("/api/talk", data={"text": "hi"}))
    assert got == [{"type": "say", "text": "Online."}, {"type": "done"}]


def test_recording_with_local_ears_is_heard_and_answered(tmp_path):
    gemini = FakeGemini([chunk(text("Roger."))])
    ears = FakeEars("what time is it")
    station = make_station(tmp_path, gemini, ears=ears)
    recording = wav_bytes([0, 16384, -16384, 0])

    response = make_client(station).post(
        "/api/talk", files={"audio": ("rec.wav", recording, "audio/wav")}
    )

    got = events(response)
    assert got[0] == {"type": "heard", "text": "what time is it"}
    assert [e["type"] for e in got] == ["heard", "say", "done"]
    assert ears.heard[0].dtype == np.float32
    assert ears.heard[0].tolist() == [0.0, 0.5, -0.5, 0.0]
    assert gemini.requests[0]["contents"][-1].parts[0].text == "what time is it"


def test_read_wav_returns_float32_at_16khz():
    samples = read_wav(wav_bytes([0, 32767, -32768]))
    assert samples.dtype == np.float32
    assert samples.tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])


def test_read_wav_resamples_to_16khz():
    tone = (np.sin(np.linspace(0, 20, 48000)) * 10000).astype("<i2")
    samples = read_wav(wav_bytes(tone, rate=48000))
    assert samples.dtype == np.float32
    assert samples.size == 16000
    assert read_wav(wav_bytes([], rate=44100)).size == 0


def test_read_wav_rejects_other_formats():
    for body in (b"not a wav at all", wav_bytes([0, 0], channels=2), wav_bytes([0], width=1)):
        with pytest.raises(web.HTTPException) as error:
            read_wav(body)
        assert error.value.status_code == 400


def test_non_wav_recording_with_local_ears_is_reported(tmp_path):
    station = make_station(tmp_path, FakeGemini(), ears=FakeEars("x"))
    response = make_client(station).post(
        "/api/talk", files={"audio": ("rec.webm", b"\x1aE\xdf\xa3junk", "audio/webm")}
    )
    assert response.status_code == 200
    got = events(response)
    assert [e["type"] for e in got] == ["error", "done"]
    assert got[0]["message"] == "That recording isn't a WAV file."


def test_recording_without_ears_is_transcribed_by_gemini(tmp_path):
    gemini = TranscribingGemini([chunk(text("Online."))], transcript=" Hello BT. ")
    station = make_station(tmp_path, gemini)
    recording = wav_bytes([1, 2, 3])

    response = make_client(station).post(
        "/api/talk", files={"audio": ("rec.wav", recording, "audio/wav")}
    )

    assert [e["type"] for e in events(response)] == ["heard", "say", "done"]
    assert events(response)[0]["text"] == "Hello BT."
    assert gemini.transcriptions[0][0].inline_data.data == recording
    assert station.memory.exchanges == [Exchange("Hello BT.", "Online.")]


@pytest.mark.parametrize("transcript", ["", "NO_SPEECH", "no_speech."])
def test_silent_recording_stops_before_asking_for_a_reply(tmp_path, transcript):
    gemini = TranscribingGemini(transcript=transcript)
    station = make_station(tmp_path, gemini)

    response = make_client(station).post(
        "/api/talk", files={"audio": ("rec.wav", wav_bytes([0, 0]), "audio/wav")}
    )

    got = events(response)
    assert [e["type"] for e in got] == ["heard", "error", "done"]
    assert got[0]["text"] == ""
    assert gemini.requests == []
    assert station.memory.exchanges == []


# --- tools ---------------------------------------------------------------------------


def timer_call(seconds=90, label="tea"):
    call = types.FunctionCall(id="t1", name="set_timer", args={"seconds": seconds, "label": label})
    return chunk(types.Part(function_call=call))


def test_set_timer_becomes_a_timer_event_with_audio(tmp_path):
    gemini = FakeGemini([timer_call()], [chunk(text("Timer set."))])
    voice = FakeVoice()
    station = make_station(tmp_path, gemini, voice=voice)

    got = events(make_client(station).post("/api/talk", data={"text": "90s tea timer"}))

    assert [e["type"] for e in got] == ["timer", "say", "done"]
    timer = got[0]
    assert (timer["seconds"], timer["label"]) == (90, "tea")
    assert timer["text"] == "Pilot, your tea is complete."
    assert timer["audio"] and timer["rate"] == 22050
    assert ("Pilot, your tea is complete.", False) in voice.calls
    response = gemini.requests[1]["contents"][-1].parts[0].function_response
    assert response.response["status"] == "set"


def test_toolbox_offers_the_desktop_tools(tmp_path):
    gemini = FakeGemini([chunk(text("Online."))])
    make_client(make_station(tmp_path, gemini)).post("/api/talk", data={"text": "hi"})

    declarations = gemini.requests[0]["config"].tools[0].function_declarations
    assert [d.name for d in declarations] == [
        "get_time",
        "system_status",
        "set_timer",
        "get_weather",
        "open_url",
        "open_app",
        "remember",
    ]


# --- errors --------------------------------------------------------------------------


def test_model_error_is_friendly_and_not_recorded(tmp_path):
    gemini = FakeGemini([RuntimeError("429 RESOURCE_EXHAUSTED")])
    station = make_station(tmp_path, gemini)

    got = events(make_client(station).post("/api/talk", data={"text": "hi"}))

    assert [e["type"] for e in got] == ["error", "done"]
    assert "capacity" in got[0]["message"]
    assert "429" not in got[0]["message"]
    assert station.memory.exchanges == []


def test_error_after_partial_reply_keeps_what_was_said_but_records_nothing(tmp_path):
    gemini = FakeGemini([chunk(text("Par. ")), RuntimeError("503 UNAVAILABLE")])
    station = make_station(tmp_path, gemini)

    got = events(make_client(station).post("/api/talk", data={"text": "hi"}))

    assert [e["type"] for e in got] == ["say", "error", "done"]
    assert station.memory.exchanges == []


def test_describe_error_variants():
    assert "overloaded" in web.describe_error(RuntimeError("503 UNAVAILABLE"))
    assert "rejected" in web.describe_error(RuntimeError("API_KEY_INVALID"))
    assert web.describe_error(RuntimeError("boom")) == "Protocol error: boom"


# --- validation ----------------------------------------------------------------------


def test_empty_text_is_rejected(tmp_path):
    client = make_client(make_station(tmp_path, FakeGemini()))
    assert client.post("/api/talk", data={"text": "   "}).status_code == 400
    assert client.post("/api/talk").status_code == 400


def test_oversized_audio_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "MAX_AUDIO_BYTES", 100)
    client = make_client(make_station(tmp_path, FakeGemini()))
    big = wav_bytes(np.zeros(200))
    response = client.post("/api/talk", files={"audio": ("rec.wav", big, "audio/wav")})
    assert response.status_code == 413


def test_long_text_is_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "MAX_TEXT_CHARS", 10)
    gemini = FakeGemini([chunk(text("Ok."))])
    make_client(make_station(tmp_path, gemini)).post("/api/talk", data={"text": "x" * 50})
    assert gemini.requests[0]["contents"][-1].parts[0].text == "x" * 10


# --- setup ---------------------------------------------------------------------------


def test_setup_needed_without_a_key(tmp_path):
    station = make_station(tmp_path, key=None)
    client = make_client(station)

    assert station.brain is None
    status = client.get("/api/status").json()
    assert (status["setup_needed"], status["brain"]) == (True, None)

    got = events(client.post("/api/talk", data={"text": "hi"}))
    assert [e["type"] for e in got] == ["error", "done"]
    assert "API key" in got[0]["message"]


@pytest.mark.parametrize("key", ["short", "has spaces in it but is long enough", "a" * 201])
def test_setup_rejects_keys_that_do_not_look_right(tmp_path, key):
    station = make_station(tmp_path, key=None)
    response = make_client(station).post("/api/setup", data={"api_key": key})
    assert response.status_code == 400
    assert station.brain is None
    assert not (tmp_path / ".env").exists()


def fake_genai(monkeypatch, valid, checked=None):
    class FakeModels:
        def get(self, model):
            if checked is not None:
                checked.append(model)
            if not valid:
                raise RuntimeError("API key not valid")

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr("google.genai.Client", FakeClient)


def test_setup_saves_the_key_and_connects(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    checked = []
    fake_genai(monkeypatch, valid=True, checked=checked)
    env = tmp_path / ".env"
    env.write_text("BT_BACKEND=gemini\nGEMINI_API_KEY=old\nBT_MODEL=x\n")
    station = make_station(tmp_path, key=None)

    response = make_client(station).post("/api/setup", data={"api_key": f" {GOOD_KEY} "})

    assert response.status_code == 200
    assert response.json()["ok"] is True and response.json()["brain"]
    assert env.read_text() == f"BT_BACKEND=gemini\nGEMINI_API_KEY={GOOD_KEY}\nBT_MODEL=x\n"
    assert env.stat().st_mode & 0o777 == 0o600
    assert station.brain is not None and not station.setup_needed
    assert os.environ["GEMINI_API_KEY"] == GOOD_KEY
    assert checked == [station.settings.model]


def test_setup_creates_env_and_appends_the_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_genai(monkeypatch, valid=True)
    env = tmp_path / ".env"
    env.write_text("OTHER=1")

    make_client(make_station(tmp_path, key=None)).post("/api/setup", data={"api_key": GOOD_KEY})

    assert env.read_text() == f"OTHER=1\nGEMINI_API_KEY={GOOD_KEY}\n"


def test_setup_with_a_key_google_rejects(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_genai(monkeypatch, valid=False)
    station = make_station(tmp_path, key=None)

    response = make_client(station).post("/api/setup", data={"api_key": GOOD_KEY})

    assert response.status_code == 400
    assert station.brain is None and station.setup_needed
    assert not (tmp_path / ".env").exists()
    assert "GEMINI_API_KEY" not in os.environ


def test_setup_is_refused_for_ollama(tmp_path):
    settings = Settings(backend="ollama", data_dir=tmp_path)
    station = Station(settings, Memory.load(None), env_path=tmp_path / ".env")
    response = make_client(station).post("/api/setup", data={"api_key": GOOD_KEY})
    assert response.status_code == 400
    assert not (tmp_path / ".env").exists()


# --- memory --------------------------------------------------------------------------


def test_memory_can_be_read_and_forgotten(tmp_path):
    memory = Memory.load(tmp_path / "memory.json")
    memory.add_fact("Pilot likes tea")
    memory.add_exchange("hi", "Online.")
    gemini = FakeGemini([chunk(text("Yes."))])
    station = make_station(tmp_path, gemini, memory=memory)
    client = make_client(station)

    assert client.get("/api/memory").json() == {"facts": ["Pilot likes tea"], "exchanges": 1}

    assert client.delete("/api/memory").json() == {"ok": True}
    assert client.get("/api/memory").json() == {"facts": [], "exchanges": 0}
    client.post("/api/talk", data={"text": "Do you remember?"})
    sent = [c.parts[0].text for c in gemini.requests[0]["contents"]]
    assert sent == ["Do you remember?"]  # the brain's own history went too


def test_non_wav_upload_is_a_400_through_read_wav():
    with pytest.raises(web.HTTPException, match="WAV"):
        read_wav(b"definitely not audio")
