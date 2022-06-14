import logging
import typing
from collections import defaultdict
from enum import Enum
from typing import List, Optional

import numpy as np
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
    NLU,
    NLUResult,
    sort_expression_ranking,
)
from dialogue_bot.models.utils import RankingScore, complete_ranking

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
        NLUTrainingData,
    )

logger = logging.getLogger(__name__)


class ExpressionUsage(Enum):
    ADD_SCORES = 1
    PERFECT_MATCH = 2
    FILTER_TOPK = 3


class NLUPipelineElement(object):
    def __init__(
        self,
        nlu: "NLU",
        use_expressions: bool = True,
        use_entities: bool = True,
        expression_usage: "ExpressionUsage" = ExpressionUsage.ADD_SCORES,
        expression_top_k: int = 15,  # only used for ExpressionUsage.FILTER_TOPK
    ):
        self.nlu = nlu
        self.use_expressions = use_expressions
        self.use_entities = use_entities
        self.expression_usage = expression_usage
        self.expression_top_k = expression_top_k

    def __repr__(self):
        return '({}: "{}", use_expressions={}, use_entities={}, expression_usage={})'.format(
            self.__class__.__name__,
            self.nlu.id,
            self.use_expressions,
            self.use_entities,
            self.expression_usage,
        )


class PipelineNLU(NLU):
    def __init__(self, env: "BotEnv", id: str, elements: List["NLUPipelineElement"]):
        super().__init__(env, id)
        self.elements = elements

    def init(self, retrain: bool, training_data: Optional["NLUTrainingData"] = None):
        logger.info(
            "Initializing {} (retrain={}) ...".format(self.__class__.__name__, retrain)
        )

        for element in self.elements:
            logger.info('Initialize NLU "{}"'.format(element.nlu.id))
            element.nlu.init(retrain, training_data=training_data)

    def run(self, utterance: str, intent_filter: List[str] = None) -> "NLUResult":
        if intent_filter is None:
            intent_filter = [intent.id for intent in self.env.intents()]
        expression_filter = [
            expression.id
            for expression in self.env.nl_expressions(intent_filter=intent_filter)
        ]

        perfect_expression = None
        expression_scores = defaultdict(list)  # expression-id => scores from nlus
        confidence_thresholds = []  # thresholds from nlus
        entities = set([])

        curr_intent_filter = intent_filter.copy()
        for element in self.elements:

            # some cases when we dont need to run the pipeline-element
            if (not element.use_expressions) and (not element.use_entities):
                logger.info('Skipping "{}"'.format(element))
                continue
            if (perfect_expression is not None) and (not element.use_entities):
                logger.info('Skipping "{}"'.format(element))
                continue

            logger.info('Running "{}"'.format(element))
            res = element.nlu.run(utterance, intent_filter=curr_intent_filter)
            res.plog()

            # expressions
            if element.use_expressions and (perfect_expression is None):

                if element.expression_usage == ExpressionUsage.ADD_SCORES:

                    # add scores
                    for s in res.expression_ranking:
                        expression_scores[s.ref_id].append(s.score)

                    # add threshold
                    confidence_thresholds.append(res.confidence_threshold)

                elif element.expression_usage == ExpressionUsage.PERFECT_MATCH:

                    # set perfect expression
                    if (len(res.expression_ranking) > 0) and (
                        res.expression_ranking[0].score >= 1
                    ):
                        perfect_expression = res.expression_ranking[0].ref_id

                elif element.expression_usage == ExpressionUsage.FILTER_TOPK:

                    # limit intents
                    expression_ids = [
                        s.ref_id
                        for s in res.expression_ranking[: element.expression_top_k]
                    ]
                    curr_intent_filter = [
                        intent.id
                        for intent in self.env.intents(
                            nl_expression_filter=expression_ids
                        )
                    ]
                else:
                    raise AssertionError(
                        'Unknown ExpressionUsage "{}"'.format(element.expression_usage)
                    )

            # entities
            if element.use_entities:
                entities.update(res.entities)

        if perfect_expression is not None:
            confidence_threshold = 1.0
            expression_ranking = [RankingScore(perfect_expression, 1.0)]
        else:
            confidence_threshold = np.mean(confidence_thresholds)
            expression_ranking = [
                RankingScore(i, float(np.mean(ss)))
                for i, ss in expression_scores.items()
            ]
            expression_ranking = sort_expression_ranking(expression_ranking)

        # complete
        expression_ranking = complete_ranking(expression_ranking, expression_filter)

        return NLUResult(utterance, expression_ranking, confidence_threshold, entities)
