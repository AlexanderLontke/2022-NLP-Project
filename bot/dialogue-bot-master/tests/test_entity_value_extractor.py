import logging

import pytest

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.intent_understanding.natural_language_understanding.tools.entity_value_extractor import (
    EntityValueExtractor,
)
from dialogue_bot.models.value import ValueSynonym

logger = logging.getLogger(__name__)


@pytest.fixture
def my_bot():
    bot = BotEnv("test", "en")

    # Values
    bot.register_value(
        "COUNTRY_DE",
        synonyms=[
            "Deutschland",
            ValueSynonym("COUNTRY_DE", "BRD", case_sensitive=True),
        ],
        regex_patterns=[r"(?i:Germa\w+)"],
    )
    bot.register_value(
        "COUNTRY_IT",
        synonyms=[
            "Italien",
            "Italy",
            ValueSynonym("COUNTRY_IT", "it", entity_context="country"),
        ],
    )
    bot.register_value(
        "PRONOUN",
        synonyms=[
            ValueSynonym("PRONOUN", "he", entity_context="pronoun"),
            ValueSynonym("PRONOUN", "she", entity_context="pronoun"),
            ValueSynonym("PRONOUN", "it", entity_context="pronoun"),
        ],
    )

    # Entities
    bot.register_entity("country", value_refs=["COUNTRY_DE", "COUNTRY_IT"])
    bot.register_entity("pronoun", value_refs=["PRONOUN"])

    bot.register_intent("country-intent", entity_filter=["country"])
    bot.register_intent_phrase_patterns(
        "country-intent",
        [
            "I live in ((country))",
            "I moved to ((country))",
            "My home country is ((country))",
            "((country)) is my home country",
        ],
    )

    bot.register_intent("pronoun-intent", entity_filter=["pronoun"])
    bot.register_intent_phrase_patterns(
        "pronoun-intent",
        [
            "My pronoun is ((pronoun))",
        ],
    )

    return bot


def test_case(my_bot):
    model = EntityValueExtractor(my_bot)
    model.init(True)

    res = model.run(None, "I live in deutschland.")
    assert len(res) == 1
    assert (res[0].entity, res[0].value) == ("country", "COUNTRY_DE")

    res = model.run("country", "I live in germany.")
    assert len(res) == 1
    assert (res[0].entity, res[0].value) == ("country", "COUNTRY_DE")

    res = model.run(None, "Do you know wht BRD is?")
    assert len(res) == 1
    assert (res[0].entity, res[0].value) == ("country", "COUNTRY_DE")

    res = model.run(None, "Do you know what brd is?")
    assert len(res) == 0

    res = model.run(None, "I live in it.")
    assert len(res) == 0

    res = model.run("country", "I live in it.")
    assert len(res) == 1
    assert (res[0].entity, res[0].value) == ("country", "COUNTRY_IT")
