import logging

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.models.pattern import PhrasePattern

logger = logging.getLogger(__name__)


def print_phrases(phrases):
    logger.info(100 * "-")
    for phrase in phrases:
        logger.info(phrase)


def test_generate():
    env = BotEnv("test", "en")

    env.register_value("COUNTRY_DE", synonyms=["DE", "Germany", "Deutschland", "BRD"])
    env.register_value("COUNTRY_CH", synonyms=["CH", "Switzerland", "Schweiz"])
    env.register_value("COUNTRY_IT", synonyms=["IT"])
    env.register_entity(
        "country", value_refs=["COUNTRY_DE", "COUNTRY_CH", "COUNTRY_IT"]
    )

    p = PhrasePattern("01", "I live in ((country))")

    phrases = list(p.generate_phrases(env, None))
    print_phrases(phrases)
    assert len(phrases) == 8

    phrases = list(p.generate_phrases(env, 5))
    print_phrases(phrases)
    assert len(phrases) == 5

    phrases = list(p.generate_phrases(env, None, max_entity_values=2))
    print_phrases(phrases)
    values = set([phrase.entities[0].value for phrase in phrases])
    assert len(values) == 2

    phrases = list(p.generate_phrases(env, None, max_value_synonyms=2))
    print_phrases(phrases)
    assert len(phrases) == 5

    phrases = list(
        p.generate_phrases(env, None, max_entity_values=2, max_value_synonyms=1)
    )
    print_phrases(phrases)
    values = set([phrase.entities[0].value for phrase in phrases])
    assert len(values) == 2
    assert len(phrases) == 2
