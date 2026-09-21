# BT-7274

An AI agent with the voice and personality of BT-7274, the Vanguard-class Titan from
Titanfall 2. Talk to him in your terminal by typing or with your voice. He answers out loud
in his own voice, can check the weather, set timers and report on your computer's status,
and remembers you between sessions.

> Trust me, Pilot.

## Quick start

You need [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS). It installs the
right Python version for you.

```bash
git clone https://github.com/FilipKlaic/BT-AI-agent.git
cd BT-AI-agent
cp .env.example .env              # then put your Gemini API key in .env
uv sync --extra listen            # leave out --extra listen to skip push-to-talk
uv run bt
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. **Use your own key** and
never commit `.env` (it's already in `.gitignore`).

The first time you use push-to-talk, macOS will ask for microphone access for your
terminal app, and the speech model (about 150 MB) is downloaded once.

## Talking to BT

| Input | What it does |
|---|---|
| Type a message | BT replies in text and voice, starting to speak after the first sentence |
| Enter on an empty line | Interrupts BT, then push-to-talk: speak, press Enter to send |
| `/talk` | Push-to-talk |
| `/hush` | Stop BT talking |
| `/voice` | Cycle Text + Voice / Text Only / Voice Only |
| `/radio` | Toggle a comms radio effect on BT's voice |
| `/memory` | Show what BT remembers about you |
| `/forget` | Wipe all pilot data |
| `/help` | List commands |
| `exit`, `quit`, `shutdown` | Power down |
| Ctrl+C while BT is replying | Cut the reply short |

Things BT can do for you (tools): tell the time, report battery/CPU/memory/disk status,
set timers that he announces out loud, look up the weather (via
[Open-Meteo](https://open-meteo.com), no key needed), open links and apps, and remember
facts you tell him.

### Options

```text
uv run bt --help
  --backend {gemini,ollama}   Language model backend
  --model MODEL               e.g. gemini-2.5-flash or llama3.2
  --text-only                 Don't load BT's voice
  --voice-only                Speak replies without printing them
  --radio                     Start with the comms radio effect on
  --no-listen                 Don't load speech recognition
  --no-memory                 Don't load or save pilot data
  --fast                      Skip the boot/shutdown animation
```

All settings can also go in `.env`; see [`.env.example`](.env.example).

### Running offline with Ollama

Install [Ollama](https://ollama.com), pull a model that supports tools, and point BT at it:

```bash
ollama pull llama3.2
uv run bt --backend ollama
```

Models without tool support still work; BT just can't use his tools with them.

## How it works

```
src/bt7274/
  __main__.py   CLI entry point and startup wiring
  config.py     Settings (from .env / environment) and BT's persona
  brain.py      Gemini and Ollama backends: streaming replies + tool calls
  tools.py      What BT can do: weather, timers, system status, ...
  memory.py     Pilot data saved to ~/.bt7274/memory.json
  speech.py     Markdown cleanup and sentence splitting for the voice
  voice.py      BT's Piper voice: synthesize and play sentences in parallel
  effects.py    Comms radio filter
  listen.py     Push-to-talk speech recognition (faster-whisper)
  terminal.py   The terminal interface
voice_model/    BT-7274 Piper voice model (ONNX)
tests/
```

Replies stream from the model into a sentence splitter. Each finished sentence is
synthesized by [Piper](https://github.com/OHF-Voice/piper1-gpl) on one thread and played
on another, so BT starts talking while the rest of the reply is still being generated.

To add a tool, write a function in `tools.py` that returns a dict and register it in
`build_toolbox()` with a JSON schema for its arguments. It then works with both backends.

## Development

```bash
uv sync --extra listen
uv run pytest          # tests use fake model backends and a fake sound card
uv run ruff check . && uv run ruff format .
```
