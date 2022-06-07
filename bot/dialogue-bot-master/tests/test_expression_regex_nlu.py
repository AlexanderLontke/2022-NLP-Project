import logging

import pytest

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.intent_understanding.natural_language_understanding.regex_er import RegexExpressionRanker
from dialogue_bot.models.state import DialogueState

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def my_bot():
    bot = BotEnv('test', 'en')

    # Values

    # Intents
    bot.register_intent('hello')
    bot.register_intent_regex_patterns('hello', [
        r'^Hel\w+$',
        r'^Hu\w+$'
    ])

    return bot

@pytest.mark.parametrize(
    "utterance, exp_conf_expression",
    [
        ("Hello", 'hello-expression'),
        ("hello", None),
        ("Hello!", None),
        ("Huhu", 'hello-expression'),
        ("Huhuuuu", 'hello-expression'),
        ("Hu Hu", None),
    ],
)
def test_nlu(my_bot, utterance, exp_conf_expression):
    nlu = RegexExpressionRanker(my_bot, 'RegexExpressionRanker')
    nlu.init(True)
    res = nlu.run(utterance)
    logger.info(res)
    if exp_conf_expression is None:
        assert res.conf_expression() is None
    else:
        assert res.conf_expression() is not None
        assert res.conf_expression().ref_id == exp_conf_expression
    assert len(res.entities) == 0
