import logging

import pytest

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.intent_understanding.natural_language_understanding.tools.entity_value_mapper import \
    EntityValueMapper
from dialogue_bot.models.value import ValueSynonym

logger = logging.getLogger(__name__)


@pytest.fixture
def my_bot():
    bot = BotEnv('test', 'en')

    # Values
    bot.register_value('COUNTRY_DE', synonyms=[
        'Deutschland',
        ValueSynonym('COUNTRY_DE', 'BRD', case_sensitive=True),
    ], regex_patterns=[
        r'(?i:Germa\w+)'
    ])
    bot.register_value('COUNTRY_IT', synonyms=[
        'Italien',
        'Italy',
        ValueSynonym('COUNTRY_IT', 'it', entity_context='country')
    ])
    bot.register_value('PRONOUN', synonyms=[
        ValueSynonym('PRONOUN', 'he', entity_context='pronoun'),
        ValueSynonym('PRONOUN', 'she', entity_context='pronoun'),
        ValueSynonym('PRONOUN', 'it', entity_context='pronoun'),
    ])

    # Entities
    bot.register_entity('country', value_refs=['COUNTRY_DE', 'COUNTRY_IT'])
    bot.register_entity('pronoun', value_refs=['PRONOUN'])

    bot.register_intent('country-intent', entity_filter=['country'])
    bot.register_intent_phrase_patterns('country-intent', [
        'I live in ((country))',
        'I moved to ((country))',
        'My home country is ((country))',
        '((country)) is my home country'])

    bot.register_intent('pronoun-intent', entity_filter=['pronoun'])
    bot.register_intent_phrase_patterns('pronoun-intent', [
        'My pronoun is ((pronoun))',
    ])

    return bot


def test_case(my_bot):
    model = EntityValueMapper(my_bot)
    model.init(True)

    res = model.run(None, 'deutschland')
    assert res is not None
    assert res == {'entity': 'country', 'value': 'COUNTRY_DE'}

    res = model.run(None, 'BRD')
    assert res is not None
    assert res == {'entity': 'country', 'value': 'COUNTRY_DE'}

    res = model.run(None, 'brd')
    assert res is None


def test_regex(my_bot):
    model = EntityValueMapper(my_bot)
    model.init(True)

    res = model.run(None, 'germany')
    assert res is not None
    assert res == {'entity': 'country', 'value': 'COUNTRY_DE'}

    res = model.run(None, 'germmany')
    assert res is None


def test_ambiguity(my_bot):
    model = EntityValueMapper(my_bot)
    model.init(True)

    res = model.run(None, 'it')
    assert res is None

    res = model.run('country', 'it')
    assert res is not None
    assert res == {'entity': 'country', 'value': 'COUNTRY_IT'}

    res = model.run(None, 'it', intent_filter=['country-intent'])
    assert res is None

    res = model.run('country', 'it', intent_filter=['country-intent'])
    assert res is not None
    assert res == {'entity': 'country', 'value': 'COUNTRY_IT'}
