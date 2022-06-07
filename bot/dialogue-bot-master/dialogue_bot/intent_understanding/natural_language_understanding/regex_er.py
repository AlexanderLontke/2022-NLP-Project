import logging
import re
import typing
from collections import defaultdict
from typing import List, Optional

from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLU, NLUResult, \
    create_nlu_train_test_data
from dialogue_bot.models.utils import complete_ranking, RankingScore

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLUTrainingData

logger = logging.getLogger(__name__)


class RegexExpressionRanker(NLU):
    """
    This NLU only ranks expressions => no entity extraction.
    This NLU uses regex-patters (include & exclude regex patterns from `NLExpression`) to identify Expressions.
    If a match was found, the expression gets a score of 1 while all others get 0.
    """

    def __init__(self, env: 'BotEnv', id: str):
        super().__init__(env, id)
        self._include_expression_patterns = defaultdict(set)  # expression-id => set(regex_patterns)
        self._exclude_expression_patterns = defaultdict(set)  # expression-id => set(regex_patterns)

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        logger.info('Initializing {} (retrain={}) ...'.format(self.__class__.__name__, retrain))

        if training_data is None:
            training_data, _ = create_nlu_train_test_data(self.env, test_size=0)

        self._include_expression_patterns = defaultdict(set)  # expression-id => set(regex_patterns)
        self._exclude_expression_patterns = defaultdict(set)  # expression-id => set(regex_patterns)
        for expression in self.env.nl_expressions(intent_filter=training_data.intent_filter):
            self._include_expression_patterns[expression.id].update(expression.regex_patterns)
            self._exclude_expression_patterns[expression.id].update(expression.exclude_regex_patterns)

    def run(self, utterance: str, intent_filter: List[str] = None) -> 'NLUResult':
        if intent_filter is None:
            intent_filter = [intent.id for intent in self.env.intents(intent_filter=intent_filter)]
        expression_filter = [expression.id for expression in self.env.nl_expressions(intent_filter=intent_filter)]

        candidate_expressions = set([])
        for expression_id in expression_filter:
            match = any([bool(re.match(pattern, utterance)) for pattern in self._include_expression_patterns[expression_id]]) \
                    and not any([bool(re.match(pattern, utterance)) for pattern in self._exclude_expression_patterns[expression_id]])
            if match:
                candidate_expressions.add(expression_id)

        # only if unambiguous
        expression_ranking = []
        if len(candidate_expressions) == 1:
            expression_ranking = [RankingScore(list(candidate_expressions)[0], 1.)]

        # complete
        expression_ranking = complete_ranking(expression_ranking, expression_filter)

        return NLUResult(utterance, expression_ranking, 1., set([]))



