"""
Text helpers for turning BT's streamed replies into speakable sentences.

Pure functions only, so they're easy to test.
"""

import re

# Spoken forms for things the voice model mispronounces. Matched on word boundaries
# so e.g. "UI" doesn't fire inside "GUI".
VOICE_REPLACEMENTS = {
    "BT-7274": "BT seventy two seventy four",
    "API": "A P I",
    "AI": "A I",
    "UI": "U I",
}

_MARKDOWN_RULES = [
    (re.compile(r"```.*?```", re.DOTALL), " "),  # code blocks aren't worth reading out
    (re.compile(r"`([^`]*)`"), r"\1"),  # `code` -> code
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"https?://\S+"), "the link"),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold** -> bold
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)"), r"\1"),  # *italic* -> italic
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),  # headings
    (re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE), ""),  # list markers
]

_REPLACEMENT_RULES = [
    (re.compile(rf"(?<![\w-]){re.escape(old)}(?![\w-])"), new)
    for old, new in VOICE_REPLACEMENTS.items()
]

# Words ending in a period that don't end a sentence.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "e.g", "i.e", "approx", "sgt", "lt", "cpt"}

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")


def clean_for_speech(text: str) -> str:
    """Strip markdown and spell out terms the voice model would mangle."""
    for pattern, repl in _MARKDOWN_RULES:
        text = pattern.sub(repl, text)
    for pattern, repl in _REPLACEMENT_RULES:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """
    Collects streamed text and hands back complete sentences as soon as they end,
    so BT can start speaking the first sentence while the rest is still arriving.
    """

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        sentences = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            candidate = self._buffer[start : match.end()].strip()
            last_word = candidate.rstrip(".!?…\"')]").rsplit(maxsplit=1)[-1:] or [""]
            if match.group().lstrip()[:1] == "." and last_word[0].lower() in _ABBREVIATIONS:
                continue
            if candidate:
                sentences.append(candidate)
            start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        """Return whatever is left once the stream has finished."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest
