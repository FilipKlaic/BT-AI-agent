import httpx
import pytest

from bt7274 import tools
from bt7274.memory import Memory


@pytest.fixture
def alerts():
    return []


def make_toolbox(alerts, handler=None):
    http = httpx.Client(transport=httpx.MockTransport(handler)) if handler else None
    return tools.build_toolbox(Memory.load(None), alerts.append, http=http)


def test_every_tool_has_a_valid_schema(alerts):
    toolbox = make_toolbox(alerts)
    assert len(toolbox) >= 6
    for tool in toolbox:
        assert tool.parameters["type"] == "object"
        assert set(tool.parameters["required"]) <= set(tool.parameters["properties"])


def test_unknown_tools_and_bad_arguments_return_errors(alerts):
    toolbox = make_toolbox(alerts)
    assert "error" in toolbox.call("self_destruct", {})
    assert "error" in toolbox.call("set_timer", {"minutes": 5})


def test_remember_stores_facts(alerts):
    memory = Memory.load(None)
    toolbox = tools.build_toolbox(memory, alerts.append)
    assert toolbox.call("remember", {"fact": "Pilot prefers tea"})["stored"] is True
    assert memory.facts == ["Pilot prefers tea"]


def test_timer_announces_when_done(alerts, monkeypatch):
    started = []

    class FakeTimer:
        def __init__(self, seconds, callback, args):
            started.append(seconds)
            self.fire = lambda: callback(*args)
            self.daemon = False

        def start(self):
            self.fire()

    monkeypatch.setattr(tools.threading, "Timer", FakeTimer)
    toolbox = make_toolbox(alerts)
    assert toolbox.call("set_timer", {"seconds": 90, "label": "tea timer"})["status"] == "set"
    assert started == [90]
    assert alerts == ["Pilot, your tea timer is complete."]
    assert "error" in toolbox.call("set_timer", {"seconds": 0})


def test_open_url_only_allows_web_links(alerts, monkeypatch):
    opened = []
    monkeypatch.setattr(tools.webbrowser, "open", lambda url: opened.append(url) or True)
    toolbox = make_toolbox(alerts)
    assert "error" in toolbox.call("open_url", {"url": "file:///etc/passwd"})
    assert toolbox.call("open_url", {"url": "https://www.ea.com/games/titanfall"}) == {
        "opened": True
    }
    assert opened == ["https://www.ea.com/games/titanfall"]


def test_system_status_reports_basics():
    status = tools.system_status()
    assert {"cpu_percent", "memory_percent", "disk_percent", "uptime_hours"} <= set(status)


def test_weather_uses_open_meteo(alerts):
    def handler(request: httpx.Request):
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "Zagreb"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "Zagreb", "country": "Croatia", "latitude": 45.8, "longitude": 16}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 21.5,
                    "apparent_temperature": 22.0,
                    "weather_code": 2,
                    "wind_speed_10m": 9.0,
                },
                "daily": {
                    "temperature_2m_max": [25.0],
                    "temperature_2m_min": [14.0],
                    "precipitation_probability_max": [10],
                },
            },
        )

    weather = make_toolbox(alerts, handler).call("get_weather", {"location": "Zagreb, Croatia"})
    assert weather["location"] == "Zagreb, Croatia"
    assert weather["conditions"] == "partly cloudy"
    assert weather["today_high_c"] == 25.0


def test_weather_for_unknown_place(alerts):
    toolbox = make_toolbox(alerts, lambda request: httpx.Response(200, json={}))
    assert "error" in toolbox.call("get_weather", {"location": "Typhon"})
