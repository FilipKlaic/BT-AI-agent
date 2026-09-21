"""
BT's brain: the language model backends.

Both backends stream the reply text as it arrives, run any tools the model asks
for, and only record an exchange once it succeeded, so errors never end up in
the conversation history.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx

from .config import PERSONA, Settings
from .memory import Memory
from .tools import Toolbox

# Upper bound on tool call round trips within a single reply.
MAX_TOOL_ROUNDS = 5
MAX_OUTPUT_TOKENS = 1024


class SetupError(Exception):
    """Configuration problem that should be shown to the user as-is."""


class Brain:
    def __init__(self, model: str, memory: Memory, toolbox: Toolbox, history_turns: int):
        self.model = model
        self.memory = memory
        self.toolbox = toolbox
        self.history_turns = history_turns

    @property
    def label(self) -> str:
        return self.model

    def system_prompt(self) -> str:
        if not self.memory.facts:
            return PERSONA
        facts = "\n".join(f"- {fact}" for fact in self.memory.facts)
        return f"{PERSONA}\n\nWhat you know about the Pilot:\n{facts}"

    def reply(self, text: str) -> Iterator[str]:
        """Stream BT's reply to `text`, one chunk of text at a time."""
        raise NotImplementedError

    def forget(self):
        """Drop the conversation history held for this session."""
        raise NotImplementedError


class GeminiBrain(Brain):
    def __init__(
        self,
        api_key: str,
        model: str,
        memory: Memory,
        toolbox: Toolbox,
        history_turns: int = 10,
        client: Any = None,
    ):
        super().__init__(model, memory, toolbox, history_turns)
        from google import genai
        from google.genai import types

        self._types = types
        self._client = client or genai.Client(api_key=api_key)
        # One list of Contents per exchange, so trimming never splits a tool call from its result.
        self._turns = [
            [self._text("user", item.pilot), self._text("model", item.bt)]
            for item in memory.exchanges[-history_turns:]
        ]
        declarations = [
            types.FunctionDeclaration(
                name=tool.name, description=tool.description, parameters_json_schema=tool.parameters
            )
            for tool in toolbox
        ]
        self._tools = [types.Tool(function_declarations=declarations)] if declarations else None

    @property
    def label(self) -> str:
        return f"Gemini {self.model.removeprefix('models/')}"

    def _text(self, role: str, text: str):
        return self._types.Content(role=role, parts=[self._types.Part.from_text(text=text)])

    def _thinking_config(self):
        # Thinking adds latency and its tokens count against max_output_tokens, which used
        # to cut replies short. BT answers in a sentence or two, so keep it off or minimal.
        name = self.model.removeprefix("models/")
        if name.startswith("gemini-2.5-flash"):
            return self._types.ThinkingConfig(thinking_budget=0)
        if name.startswith("gemini-3"):
            return self._types.ThinkingConfig(thinking_level=self._types.ThinkingLevel.LOW)
        return None

    def _config(self):
        return self._types.GenerateContentConfig(
            system_instruction=self.system_prompt(),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
            top_p=0.8,
            tools=self._tools,
            thinking_config=self._thinking_config(),
        )

    def reply(self, text: str) -> Iterator[str]:
        types = self._types
        history = [content for turn in self._turns[-self.history_turns :] for content in turn]
        turn = [self._text("user", text)]
        spoken: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            parts, calls = [], []
            stream = self._client.models.generate_content_stream(
                model=self.model, contents=history + turn, config=self._config()
            )
            for chunk in stream:
                content = chunk.candidates[0].content if chunk.candidates else None
                for part in (content.parts if content else None) or []:
                    # Keep every part as-is: some carry thought signatures the API wants back.
                    parts.append(part)
                    if part.function_call:
                        calls.append(part.function_call)
                    elif part.text and not part.thought:
                        spoken.append(part.text)
                        yield part.text
            if parts:
                turn.append(types.Content(role="model", parts=parts))
            if not calls:
                break
            results = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=self.toolbox.call(call.name, call.args)
                    )
                )
                for call in calls
            ]
            turn.append(types.Content(role="user", parts=results))

        reply = "".join(spoken).strip()
        if reply and turn[-1].role == "model":
            self._turns.append(turn)
            self.memory.add_exchange(text, reply)

    def forget(self):
        self._turns.clear()


class OllamaBrain(Brain):
    """Local models through Ollama (https://ollama.com), for running BT offline."""

    def __init__(
        self,
        host: str,
        model: str,
        memory: Memory,
        toolbox: Toolbox,
        history_turns: int = 10,
        http: httpx.Client | None = None,
    ):
        super().__init__(model, memory, toolbox, history_turns)
        self.host = host
        self._http = http or httpx.Client(base_url=host, timeout=httpx.Timeout(120, connect=5))
        self._turns: list[list[dict[str, Any]]] = [
            [{"role": "user", "content": item.pilot}, {"role": "assistant", "content": item.bt}]
            for item in memory.exchanges[-history_turns:]
        ]
        self._tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in toolbox
        ]
        # Not every local model supports tools; switched off on the first refusal.
        self._use_tools = bool(self._tool_specs)

    @property
    def label(self) -> str:
        return f"Ollama {self.model}"

    def reply(self, text: str) -> Iterator[str]:
        history = [message for turn in self._turns[-self.history_turns :] for message in turn]
        turn: list[dict[str, Any]] = [{"role": "user", "content": text}]
        spoken: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            messages = [{"role": "system", "content": self.system_prompt()}, *history, *turn]
            content, calls = "", []
            for message in self._stream(messages):
                if message.get("content"):
                    content += message["content"]
                    spoken.append(message["content"])
                    yield message["content"]
                calls.extend(message.get("tool_calls") or [])
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = calls
            turn.append(assistant)
            if not calls:
                break
            for call in calls:
                function = call["function"]
                result = self.toolbox.call(function["name"], function.get("arguments"))
                turn.append(
                    {"role": "tool", "tool_name": function["name"], "content": json.dumps(result)}
                )

        reply = "".join(spoken).strip()
        if reply and turn[-1]["role"] == "assistant":
            self._turns.append(turn)
            self.memory.add_exchange(text, reply)

    def _stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.7, "top_p": 0.8},
        }
        if self._use_tools:
            payload["tools"] = self._tool_specs
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode(errors="replace")
                    if self._use_tools and "does not support tools" in body:
                        self._use_tools = False
                        yield from self._stream(messages)
                        return
                    raise RuntimeError(f"Ollama returned {response.status_code}: {body.strip()}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise RuntimeError(f"Ollama: {data['error']}")
                    yield data.get("message") or {}
                    if data.get("done"):
                        return
        except httpx.ConnectError as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running (`ollama serve`)?"
            ) from error

    def forget(self):
        self._turns.clear()


def create_brain(settings: Settings, memory: Memory, toolbox: Toolbox) -> Brain:
    if settings.backend == "gemini":
        if not settings.gemini_api_key:
            raise SetupError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://aistudio.google.com/apikey), or run with --backend ollama."
            )
        return GeminiBrain(
            settings.gemini_api_key, settings.model, memory, toolbox, settings.history_turns
        )
    if settings.backend == "ollama":
        return OllamaBrain(
            settings.ollama_host, settings.model, memory, toolbox, settings.history_turns
        )
    raise SetupError(f"Unknown backend '{settings.backend}'. Use 'gemini' or 'ollama'.")
