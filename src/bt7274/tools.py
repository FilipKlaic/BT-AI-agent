"""
Tools BT can use during a conversation.

Each tool is a plain function plus a JSON schema describing its arguments, which
works for both Gemini and Ollama function calling. To add a tool, write the
function and add a `Tool(...)` entry in `build_toolbox`.
"""

import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any

import httpx
import psutil

from .memory import Memory

MAX_TIMER_SECONDS = 24 * 60 * 60

# https://open-meteo.com/en/docs (WMO weather interpretation codes)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments
    func: Callable[..., dict[str, Any]]


class Toolbox:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)

    def call(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Run a tool. Errors are returned to the model rather than raised, so BT can explain."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool.func(**(args or {}))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}


def _schema(properties: dict[str, dict] | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties or {}, "required": required or []}


def get_time(tz: tzinfo | None = None) -> dict[str, Any]:
    now = datetime.now(tz) if tz else datetime.now().astimezone()
    return {
        "local_time": now.strftime("%H:%M"),
        "date": now.strftime("%A, %d %B %Y"),
        "timezone": now.tzname(),
    }


def system_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }
    battery = psutil.sensors_battery()
    if battery is not None:
        status["battery_percent"] = round(battery.percent)
        status["charging"] = bool(battery.power_plugged)
    return status


def open_url(url: str) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        return {"error": "Only http and https links can be opened."}
    return {"opened": webbrowser.open(url)}


def open_app(name: str) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"error": "Opening apps is only supported on macOS."}
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"Could not open {name}."}
    return {"opened": name}


def get_weather(http: httpx.Client, location: str) -> dict[str, Any]:
    # The geocoder matches place names only, so drop any ", Country" suffix.
    place = location.split(",")[0].strip()
    found = http.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1, "language": "en", "format": "json"},
    )
    found.raise_for_status()
    results = found.json().get("results")
    if not results:
        return {"error": f"Could not find a place called {location}."}
    spot = results[0]
    forecast = http.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": spot["latitude"],
            "longitude": spot["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 1,
            "timezone": "auto",
        },
    )
    forecast.raise_for_status()
    data = forecast.json()
    current, daily = data["current"], data["daily"]
    return {
        "location": f"{spot['name']}, {spot.get('country', '')}".strip(", "),
        "conditions": WEATHER_CODES.get(current["weather_code"], "unknown"),
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "wind_kmh": current["wind_speed_10m"],
        "today_high_c": daily["temperature_2m_max"][0],
        "today_low_c": daily["temperature_2m_min"][0],
        "chance_of_rain_percent": daily["precipitation_probability_max"][0],
    }


# --- tool definitions ------------------------------------------------------------
# One function per tool, so the terminal and the web app can each pick their set.


def time_tool(tz: tzinfo | None = None) -> Tool:
    return Tool("get_time", "Get the current local date and time.", _schema(), lambda: get_time(tz))


def system_status_tool() -> Tool:
    return Tool(
        "system_status",
        "Get the status of the Pilot's computer: battery, CPU, memory, disk, uptime.",
        _schema(),
        system_status,
    )


def timer_tool(start: Callable[[int, str], None], tz: tzinfo | None = None) -> Tool:
    """`start(seconds, label)` arranges for BT to announce the timer when it finishes."""

    def set_timer(seconds: int, label: str = "timer") -> dict[str, Any]:
        seconds = int(seconds)
        if not 0 < seconds <= MAX_TIMER_SECONDS:
            return {"error": f"Timer must be between 1 second and {MAX_TIMER_SECONDS} seconds."}
        start(seconds, label)
        fires_at = (datetime.now(tz) if tz else datetime.now()) + timedelta(seconds=seconds)
        return {"status": "set", "label": label, "fires_at": fires_at.strftime("%H:%M:%S")}

    return Tool(
        "set_timer",
        "Set a countdown timer. BT announces it out loud when it finishes.",
        _schema(
            {
                "seconds": {"type": "integer", "description": "Duration in seconds."},
                "label": {"type": "string", "description": "Short name, e.g. 'tea'."},
            },
            required=["seconds"],
        ),
        set_timer,
    )


def weather_tool(http: httpx.Client) -> Tool:
    return Tool(
        "get_weather",
        "Get current weather and today's forecast for a city.",
        _schema(
            {"location": {"type": "string", "description": "City name, e.g. 'Zagreb'."}},
            required=["location"],
        ),
        lambda location: get_weather(http, location),
    )


def open_url_tool(opener: Callable[[str], dict[str, Any]] = open_url) -> Tool:
    return Tool(
        "open_url",
        "Open a web page in the Pilot's browser.",
        _schema({"url": {"type": "string", "description": "Full http(s) URL."}}, ["url"]),
        opener,
    )


def open_app_tool() -> Tool:
    return Tool(
        "open_app",
        "Open an application on the Pilot's Mac by name, e.g. 'Spotify'.",
        _schema({"name": {"type": "string", "description": "Application name."}}, ["name"]),
        open_app,
    )


def remember_tool(memory: Memory) -> Tool:
    def remember(fact: str) -> dict[str, Any]:
        return {"stored": memory.add_fact(fact), "fact": fact}

    return Tool(
        "remember",
        "Permanently remember a fact about the Pilot (name, preferences, plans).",
        _schema(
            {"fact": {"type": "string", "description": "One short fact in third person."}},
            required=["fact"],
        ),
        remember,
    )


def message_for_timer(label: str) -> str:
    return f"Pilot, your {label} is complete."


def build_toolbox(
    memory: Memory,
    on_alert: Callable[[str], None],
    http: httpx.Client | None = None,
) -> Toolbox:
    """Create BT's terminal tools. `on_alert` is called from a timer thread when one fires."""
    http = http or httpx.Client(timeout=10)

    def start_timer(seconds: int, label: str):
        timer = threading.Timer(seconds, on_alert, args=(message_for_timer(label),))
        timer.daemon = True
        timer.start()

    return Toolbox(
        [
            time_tool(),
            system_status_tool(),
            timer_tool(start_timer),
            weather_tool(http),
            open_url_tool(),
            open_app_tool(),
            remember_tool(memory),
        ]
    )
