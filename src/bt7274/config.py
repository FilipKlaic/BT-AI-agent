"""
Configuration and text constants for BT-7274
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.5-flash"

# System Instructions
BT_INSTRUCTION = (
    "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
    "You are logical, loyal, and speak with military precision and dry wit. "
    "Always refer to the user as 'Pilot'. Respond in-character as BT-7274. "
    "Keep responses brief and concise - 1-2 sentences maximum unless specifically asked for detail."
)

# Text Processing
VOICE_REPLACEMENTS = {
    'BT-7274': 'BT seventy two seventy four',
    'AI': 'A I',
    'API': 'A P I',
    'UI': 'U I'
}

# UI text
STARTUP_MESSAGES = [
    "═══════════════════════════════════════════════",
    "    BT-7274 VANGUARD-CLASS TITAN ONLINE",
    "═══════════════════════════════════════════════",
    "",
    "┌─ SYSTEM STATUS ─────────────────────────────┐",
    "│ Chassis: Operational                        │",
    "│ Neural Link: Established                    │", 
    "│ Weapons Systems: Standby                    │",
    # Voice status line is dynamic and handled in main.py
    "│ Pilot Interface: Active                     │",
    "└─────────────────────────────────────────────┘",
    "",
    "Hello, Pilot. I am BT-7274.",
    "Mission parameters: Assist and protect Pilot.",
    "Ready for orders.",
    "",
    "Commands:",
    "  - Type normally to chat with BT",
    # Voice commands are dynamic
    "  - 'exit', 'quit', or 'shutdown' to terminate",
    "─────────────────────────────────────────────────",
    ""
]

SHUTDOWN_MESSAGES = [
    "",
    "─────────────────────────────────────────────────",
    "Initiating shutdown sequence...",
    "Saving pilot data... Complete.",
    "Neural link terminated.",
    "Until next time, Pilot.",
    "BT-7274 systems offline.",
    "═══════════════════════════════════════════════"
]
