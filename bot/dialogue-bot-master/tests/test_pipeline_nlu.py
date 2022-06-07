import logging
from typing import List, Optional

import pytest

from dialogue_bot.bot_env import BotEnv
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLU, NLUResult, NLUTrainingData
from dialogue_bot.intent_understanding.natural_language_understanding.pipeline_nlu import PipelineNLU, \
    NLUPipelineElement, ExpressionUsage
from dialogue_bot.models.expression import NLExpression
from dialogue_bot.models.state import DialogueState
from dialogue_bot.models.triggers.nl import PhraseNLTrigger
from dialogue_bot.models.utils import RankingScore

logger = logging.getLogger(__name__)


class NLU1(NLU):

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        pass

    def run(self, utterance: str, dialogue_state: 'DialogueState', intent_filter: List[str] = None) -> 'NLUResult':
        return NLUResult(utterance, [
            RankingScore('c', 1.),
            RankingScore('a', 0.),
            RankingScore('b', 0.),
            RankingScore('d', 0.),
        ], 1., set([]))


class NLU2(NLU):

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        pass

    def run(self, utterance: str, dialogue_state: 'DialogueState', intent_filter: List[str] = None) -> 'NLUResult':
        return NLUResult(utterance, [
            RankingScore('a', 0.8),
            RankingScore('b', 0.7),
            RankingScore('c', 0.4),
            RankingScore('d', 0.3),
        ], 1., set([]))


@pytest.fixture(scope='module')
def my_bot():
    bot = BotEnv('test', 'en')

    # Intents
    a_ex = NLExpression(bot, 'a')
    b_ex = NLExpression(bot, 'b')
    c_ex = NLExpression(bot, 'c')
    d_ex = NLExpression(bot, 'd')
    bot.register_intent('A', nl_trigger=PhraseNLTrigger(a_ex))
    bot.register_intent('B', nl_trigger=PhraseNLTrigger(b_ex))
    bot.register_intent('C', nl_trigger=PhraseNLTrigger(c_ex))
    bot.register_intent('D', nl_trigger=PhraseNLTrigger(d_ex))

    return bot


def test_nlu(my_bot):
    nlu = PipelineNLU(my_bot, 'PipelineNLU', [
        NLUPipelineElement(NLU1(my_bot, 'NLU1'), expression_usage=ExpressionUsage.PERFECT_MATCH),
        NLUPipelineElement(NLU2(my_bot, 'NLU2')),
    ])
    nlu.init(True, training_data=NLUTrainingData(['a', 'b', 'c', 'd'], []))
    res = nlu.run('test')
    logger.info(res)
    logger.info(res.expression_ranking)
    assert res.expression.ref_id == 'c'
    assert res.expression.score == 1.
    assert all([s.score == 0 for s in res.expression_ranking][1:])
