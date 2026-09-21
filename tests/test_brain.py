import json

import httpx
import pytest
from google.genai import types

from bt7274.brain import GeminiBrain, OllamaBrain, SetupError, create_brain
from bt7274.config import Settings
from bt7274.memory import Exchange, Memory
from bt7274.tools import Tool, Toolbox


def echo_toolbox(calls):
    def lookup(code: str):
        calls.append(code)
        return {"answer": f"code {code} is Protocol 3"}

    schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    return Toolbox([Tool("lookup", "Look up a protocol.", schema, lookup)])


# --- Gemini --------------------------------------------------------------------------


def chunk(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def text(value):
    return types.Part.from_text(text=value)


class FakeGemini:
    """Stands in for genai.Client; each request pops the next scripted stream."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests = []
        self.models = self

    def generate_content_stream(self, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        script = self.streams.pop(0)

        def generate():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return generate()


def gemini(client, memory=None, toolbox=None, **kwargs):
    return GeminiBrain(
        "unused",
        "gemini-2.5-flash",
        memory or Memory.load(None),
        toolbox or Toolbox([]),
        client=client,
        **kwargs,
    )


def test_gemini_streams_and_records_the_exchange():
    client = FakeGemini([chunk(text("Affirmative, ")), chunk(text("Pilot."))])
    memory = Memory.load(None)
    brain = gemini(client, memory)

    assert list(brain.reply("Status?")) == ["Affirmative, ", "Pilot."]
    assert memory.exchanges == [Exchange("Status?", "Affirmative, Pilot.")]
    config = client.requests[0]["config"]
    assert config.thinking_config.thinking_budget == 0
    assert "BT-7274" in config.system_instruction


def test_gemini_runs_tools_and_sends_results_back():
    calls = []
    client = FakeGemini(
        [
            chunk(
                types.Part(
                    function_call=types.FunctionCall(id="c1", name="lookup", args={"code": "3"})
                )
            )
        ],
        [chunk(text("Protocol 3: protect the Pilot."))],
    )
    brain = gemini(client, toolbox=echo_toolbox(calls))

    assert "".join(brain.reply("What is protocol 3?")) == "Protocol 3: protect the Pilot."
    assert calls == ["3"]
    follow_up = client.requests[1]["contents"]
    response = follow_up[-1].parts[0].function_response
    assert (response.id, response.name) == ("c1", "lookup")
    assert response.response == {"answer": "code 3 is Protocol 3"}
    declaration = client.requests[0]["config"].tools[0].function_declarations[0]
    assert declaration.name == "lookup"


def test_gemini_errors_do_not_pollute_history():
    client = FakeGemini(
        [chunk(text("Par")), RuntimeError("503 overloaded")],
        [chunk(text("Online."))],
    )
    memory = Memory.load(None)
    brain = gemini(client, memory)

    with pytest.raises(RuntimeError):
        list(brain.reply("First"))
    assert memory.exchanges == []

    list(brain.reply("Second"))
    sent = [c.parts[0].text for c in client.requests[1]["contents"]]
    assert sent == ["Second"]


def test_gemini_loads_and_trims_history():
    memory = Memory.load(None)
    memory.exchanges = [Exchange(f"q{i}", f"a{i}") for i in range(5)]
    memory.facts = ["Pilot's name is Jack Cooper"]
    client = FakeGemini([chunk(text("Yes."))])
    brain = gemini(client, memory, history_turns=2)

    list(brain.reply("Remember me?"))
    sent = [c.parts[0].text for c in client.requests[0]["contents"]]
    assert sent == ["q3", "a3", "q4", "a4", "Remember me?"]
    assert "Jack Cooper" in client.requests[0]["config"].system_instruction


def test_gemini_empty_reply_is_not_recorded():
    memory = Memory.load(None)
    brain = gemini(FakeGemini([types.GenerateContentResponse(candidates=[])]), memory)
    assert list(brain.reply("Hello?")) == []
    assert memory.exchanges == []


# --- Ollama --------------------------------------------------------------------------


def ndjson(*messages):
    lines = [json.dumps({"message": m, "done": False}) for m in messages]
    lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
    return httpx.Response(200, content="\n".join(lines).encode())


def ollama(handler, memory=None, toolbox=None):
    http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaBrain(
        "http://ollama", "llama3.2", memory or Memory.load(None), toolbox or Toolbox([]), http=http
    )


def test_ollama_streams_and_runs_tools():
    calls, requests = [], []
    responses = [
        ndjson(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "lookup", "arguments": {"code": "3"}}}],
            }
        ),
        ndjson(
            {"role": "assistant", "content": "Protect "},
            {"role": "assistant", "content": "the Pilot."},
        ),
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        return responses.pop(0)

    memory = Memory.load(None)
    brain = ollama(handler, memory, echo_toolbox(calls))
    assert list(brain.reply("Protocol 3?")) == ["Protect ", "the Pilot."]
    assert calls == ["3"]
    tool_message = requests[1]["messages"][-1]
    assert tool_message == {
        "role": "tool",
        "tool_name": "lookup",
        "content": '{"answer": "code 3 is Protocol 3"}',
    }
    assert requests[0]["messages"][0]["role"] == "system"
    assert memory.exchanges == [Exchange("Protocol 3?", "Protect the Pilot.")]


def test_ollama_retries_without_tools_for_models_that_lack_them():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "tools" in body:
            return httpx.Response(400, json={"error": "llama2 does not support tools"})
        return ndjson({"role": "assistant", "content": "Roger."})

    brain = ollama(handler, toolbox=echo_toolbox([]))
    assert list(brain.reply("Hi")) == ["Roger."]
    assert list(brain.reply("Again")) == ["Roger."]
    assert ["tools" in r for r in requests] == [True, False, False]


def test_ollama_connection_error_is_readable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(RuntimeError, match="ollama serve"):
        list(ollama(handler).reply("Hi"))


# --- setup ---------------------------------------------------------------------------


def test_missing_api_key_explains_what_to_do():
    with pytest.raises(SetupError, match="GEMINI_API_KEY"):
        create_brain(Settings(gemini_api_key=None), Memory.load(None), Toolbox([]))
    with pytest.raises(SetupError, match="Unknown backend"):
        create_brain(Settings(backend="skynet"), Memory.load(None), Toolbox([]))
