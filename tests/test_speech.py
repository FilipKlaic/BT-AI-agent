from bt7274.speech import SentenceSplitter, clean_for_speech


def test_strips_markdown():
    text = "**Warning**: check `config.py` and *hurry*.\n- item one\n# Heading"
    assert clean_for_speech(text) == "Warning: check config.py and hurry. item one Heading"


def test_links_are_not_read_out():
    assert clean_for_speech("See [the docs](https://x.io) or https://y.io/z") == (
        "See the docs or the link"
    )


def test_replacements_respect_word_boundaries():
    assert clean_for_speech("I am BT-7274, an AI.") == "I am BT seventy two seventy four, an A I."
    assert clean_for_speech("The API and UI") == "The A P I and U I"
    assert clean_for_speech("A GUI, SAID the MAIN log") == "A GUI, SAID the MAIN log"


def test_splitter_emits_sentences_as_they_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("Affirmative, Pi") == []
    assert splitter.feed("lot. Protocol three") == ["Affirmative, Pilot."]
    assert splitter.feed(": protect the Pilot! Trust") == ["Protocol three: protect the Pilot!"]
    assert splitter.flush() == "Trust"
    assert splitter.flush() == ""


def test_splitter_skips_abbreviations_and_decimals():
    splitter = SentenceSplitter()
    assert splitter.feed("Ask Dr. Graves about e.g. the 3.5 ton load. Next") == [
        "Ask Dr. Graves about e.g. the 3.5 ton load."
    ]


def test_splitter_breaks_on_newlines():
    splitter = SentenceSplitter()
    assert splitter.feed("Status report\n\nAll systems nominal") == ["Status report"]
    assert splitter.flush() == "All systems nominal"
