import logging

import pytest

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLU
from dialogue_bot.intent_understanding.natural_language_understanding.phrase_er import PhraseExpressionRanker
from dialogue_bot.intent_understanding.natural_language_understanding.rasa_nlu.rasa_nlu import RasaNLU
from dialogue_bot.models.state import DialogueState
from dialogue_bot.models.value import ValueSynonym
from dialogue_bot.nlp.text_scorer.vector import VectorTextScorer
from dialogue_bot.nlp.tokenizer.nlt import NLTKWordTokenizer
from dialogue_bot.nlp.vectorizer.text.agg import AggTextVectorizer
from dialogue_bot.nlp.vectorizer.word.gensi import GensimVectorizer

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
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

    # Intents
    bot.register_intent('hello')
    bot.register_intent_phrase_patterns('hello', [
        'Hi', 'hello', 'ahoi', 'good morning', 'good evening', 'good afternoon', 'hi there'])

    bot.register_intent('bye')
    bot.register_intent_phrase_patterns('bye', [
        'Good bye', 'bye', 'see you', 'until later', 'see you later', 'I have to go'])

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


@pytest.fixture(scope='module')
def my_nlus(my_bot):
    nlus = {
        'RasaNLU': RasaNLU(my_bot, 'RasaNLU'),
        'PhraseExpressionRanker': PhraseExpressionRanker(my_bot, 'PhraseExpressionRanker.phrase',
                                           VectorTextScorer('PhraseExpressionRanker.phrase.scorer',
                                                            AggTextVectorizer('PhraseExpressionRanker.phrase.scorer.textvec',
                                                                              NLTKWordTokenizer(
                                                                                  'PhraseExpressionRanker.phrase.scorer.textvec.wordvec'),
                                                                              GensimVectorizer(
                                                                                  'PhraseExpressionRanker.phrase.scorer.textvec.wordtok')
                                                                              ),
                                                            True
                                                            ),
                                           )
    }

    for nlu in nlus.values():
        nlu.init(True)

    return nlus


@pytest.mark.parametrize("nlu_name", [
    'RasaNLU',
    'PhraseExpressionRanker'
])
@pytest.mark.parametrize(
    "utterance, exp_conf_expression, exp_n_entities, exp_entity_id, exp_entity_value",
    [
        ("Hi!", 'hello-expression', 0, None, None),
        ("Heyy", 'hello-expression', 0, None, None),
        ("helllo", 'hello-expression', 0, None, None),
        ("helllo", 'hello-expression', 0, None, None),
        ("see you tomorrow", 'bye-expression', 0, None, None),
        ("bye bye", 'bye-expression', 0, None, None),
        ('i have to go now', 'bye-expression', 0, None, None),
        ("How are you?", None, 0, None, None),
        ("My name is Matthias", None, 0, None, None),
        ("Where can I buy a car?", None, 0, None, None),
        ("I like you", None, 0, None, None),
        ("sdaf dsfds grsffd", None, 0, None, None),
        ("Currently, i live in the BRD", 'country-intent-expression', 1, 'country', 'COUNTRY_DE'),
        ("I am living in Germany", 'country-intent-expression', 1, 'country', 'COUNTRY_DE'),
        ("i live in deutschland", 'country-intent-expression', 1, 'country', 'COUNTRY_DE'),
        ("i enjoy living in italy", 'country-intent-expression', 1, 'country', 'COUNTRY_IT'),
        ("I live in italy", 'country-intent-expression', 1, 'country', 'COUNTRY_IT'),
        ("Currently, I live in italy", 'country-intent-expression', 1, 'country', 'COUNTRY_IT'),
        ("Italien is where I live", 'country-intent-expression', 1, 'country', 'COUNTRY_IT'),
    ],
)
def test_nlu(my_nlus, nlu_name, utterance, exp_conf_expression, exp_n_entities, exp_entity_id, exp_entity_value):
    nlu: 'NLU' = my_nlus[nlu_name]

    logger.info('Testing "{}"'.format(nlu_name))
    res = nlu.run(utterance)
    logger.info('NLU-Result: {}'.format(res))

    if exp_conf_expression is None:
        assert res.conf_expression() is None
    else:
        assert res.conf_expression() is not None
        assert res.conf_expression().ref_id == exp_conf_expression
    assert len(res.entities) == exp_n_entities
    if len(res.entities) == 1:
        assert list(res.entities)[0].entity == exp_entity_id
        assert list(res.entities)[0].value == exp_entity_value
