import logging
import os
import time
import typing
from abc import ABC, abstractmethod
from typing import Set, Optional, List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import plotly.express as px
from dialogue_bot import logcolor
from dialogue_bot.models.utils import stratify_train_test_split
from dialogue_bot.utils.rand import uuid_str
from sklearn.metrics import precision_score, recall_score

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.models.pattern import PhrasePattern
    from dialogue_bot.intent_understanding.iu import ExtractedEntity
    from dialogue_bot.models.utils import RankingScore

logger = logging.getLogger(__name__)


def sort_expression_ranking(
    expression_ranking: List["RankingScore"],
) -> List["RankingScore"]:
    # sort primarily by score, then by number of input contexts
    sort_key = lambda s: s.score
    return sorted(expression_ranking, key=sort_key, reverse=True)


def create_nlu_train_test_data(
    env: "BotEnv",
    intent_filter: List[str] = None,
    test_size: float = 0.0,
) -> Tuple["NLUTrainingData", "NLUTestData"]:
    if intent_filter is None:
        intent_filter = [intent.id for intent in env.intents()]

    # collect phrase-patterns
    phrase_patterns = set([])
    for expression in env.nl_expressions(intent_filter=intent_filter):
        for phrase_pattern in expression.phrase_patterns:
            phrase_patterns.add(phrase_pattern)
    phrase_patterns = list(phrase_patterns)

    # split into train/test
    train_phrase_patterns, test_phrase_patterns = stratify_train_test_split(
        phrase_patterns, lambda x: x.expression_id, label_test_split=test_size
    )

    return NLUTrainingData(intent_filter, train_phrase_patterns), NLUTestData(
        test_phrase_patterns
    )


class EvaluationResult(object):
    OTHER_CLASSNAME = "__OTHER__"

    def __init__(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold
        self.true_expressions = []
        self.pred_expressions = []
        self.pred_times = []

    def add(self, true_expression, predicted_expression, prediction_time):
        self.true_expressions.append(true_expression)
        self.pred_expressions.append(predicted_expression)
        self.pred_times.append(prediction_time)

    def avg_pred_time(self):
        return np.mean(self.pred_times)

    def precision(self, average="macro"):
        return precision_score(
            [
                t
                for t, p in zip(self.true_expressions, self.pred_expressions)
                if p != self.OTHER_CLASSNAME
            ],
            [
                p
                for t, p in zip(self.true_expressions, self.pred_expressions)
                if p != self.OTHER_CLASSNAME
            ],
            average=average,
        )

    def recall(self, average="macro"):
        return recall_score(
            self.true_expressions, self.pred_expressions, average=average
        )

    def nr_samples(self):
        return len(self.true_expressions)


class NLU(ABC):
    """
    Natural-Language-Understanding unit (NLU).
    Given a user utterance, this ranks expressions and extracts entities.
    """

    def __init__(self, env: "BotEnv", id: str):
        self.env = env
        self.id = id

    @property
    def storage_dir(self) -> str:
        return os.path.join(self.env.storage_dir, "nlu", self.id)

    @abstractmethod
    def init(self, retrain: bool, training_data: Optional["NLUTrainingData"] = None):
        """
        Should be executed before NLU is used. This will either load NLU with all of its models (when retrain==False)
        or retrain all of its models (when retrain==True)
        """
        pass

    @abstractmethod
    def run(self, utterance: str, intent_filter: List[str] = None) -> "NLUResult":
        pass

    def _evaluate(
        self,
        test_data: "NLUTestData",
        confidence_thresholds: List[float] = np.linspace(0, 1, 100),
    ) -> Tuple["EvaluationResult", List["EvaluationResult"]]:

        logger.info("Test on test-data: {}...".format(test_data))

        default_res = EvaluationResult(
            None
        )  # the result for the confidence-threshold determined by nlu
        threshold_res = [EvaluationResult(t) for t in confidence_thresholds]

        for phrase_pattern in test_data.phrase_patterns:

            for test_phrase in phrase_pattern.generate_phrases(self.env, 1):

                logger.info("Test Phrase: {}".format(test_phrase))
                start_time = time.time()
                nlu_res = self.run(test_phrase.text)
                end_time = time.time()

                top_expression = (
                    nlu_res.expression_ranking[0]
                    if len(nlu_res.expression_ranking) > 0
                    else None
                )

                logger.info("-" * 100)
                logger.info(
                    "\tExpected expression:  {}".format(test_phrase.expression_id)
                )
                logger.info(
                    "\tPredicted expression: {} (score={})".format(
                        top_expression.ref_id, top_expression.score
                    )
                )
                logger.info("-" * 100)

                # check for confidence-threshold determined by nlu
                conf_expression = nlu_res.conf_expression()
                default_res.add(
                    test_phrase.expression_id,
                    conf_expression.ref_id
                    if conf_expression is not None
                    else EvaluationResult.OTHER_CLASSNAME,
                    end_time - start_time,
                )

                # check for different confidence-thresholds
                for i, confidence_threshold in enumerate(confidence_thresholds):
                    conf_expression = nlu_res.conf_expression(
                        confidence_threshold=confidence_threshold
                    )
                    threshold_res[i].add(
                        test_phrase.expression_id,
                        conf_expression.ref_id
                        if conf_expression is not None
                        else EvaluationResult.OTHER_CLASSNAME,
                        end_time - start_time,
                    )

        return default_res, threshold_res

    def evaluate(
        self,
        train_data: "NLUTrainingData",
        test_data: Union["NLUTestData", Dict[str, "NLUTestData"]],
        confidence_thresholds: List[float] = np.linspace(0, 1, 100),
        evaluation_id: str = None,
    ):
        """
        Evaluates the NLU.
        TODO: currently, only the expression-ranking is evaluated
        :param train_data: The training data
        :param test_data:  a named dictionary of different Test-Data
        :param confidence_thresholds: The confidence thresholds to test.
        :param evaluation_id: A unique id for this evaluation.
        :return:
        """
        if evaluation_id is None:
            evaluation_id = uuid_str(6)
        if isinstance(test_data, NLUTestData):
            test_data = {"all": test_data}

        logger.info("Evaluating NLU {}...".format(self.id))

        logger.info("Train on training-data: {}".format(train_data))
        self.init(True, train_data)

        res = {}
        for t_name, t_data in test_data.items():
            logger.info('Evaluate on test-data "{}": {}'.format(t_name, t_data))
            d_res, t_ress = self._evaluate(t_data, confidence_thresholds)
            res[t_name] = (d_res, t_ress)

        # Create PR-Plot
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        pr_filepath = os.path.join(self.storage_dir, "pr_{}.html".format(evaluation_id))

        df = pd.DataFrame(columns=["test_name", "threshold", "precision", "recall"])
        avg_pred_time = []
        for t_name, (d_res, t_ress) in res.items():
            df = df.append(
                {
                    "test_name": t_name,
                    "threshold": d_res.confidence_threshold,
                    "precision": d_res.precision(),
                    "recall": d_res.recall(),
                    "nr_samples": d_res.nr_samples(),
                    "avg_pred_time": d_res.avg_pred_time(),
                    "text": "CURRENT",
                },
                ignore_index=True,
            )
            avg_pred_time.append(d_res.avg_pred_time())

            for t_res in t_ress:
                df = df.append(
                    {
                        "test_name": t_name,
                        "threshold": t_res.confidence_threshold,
                        "precision": t_res.precision(),
                        "recall": t_res.recall(),
                        "nr_samples": t_res.nr_samples(),
                        "avg_pred_time": t_res.avg_pred_time(),
                        "text": "",
                    },
                    ignore_index=True,
                )
        avg_pred_time = np.mean(avg_pred_time)

        fig = px.scatter(
            df,
            x="recall",
            y="precision",
            text="text",
            color="test_name",
            hover_data=["test_name", "threshold", "nr_samples", "avg_pred_time"],
        )
        fig.update_layout(
            title='PR-Plot for NLU "<b>{}</b>":<br>'
            "<br><b>eval-id</b>              = <b>{}</b>"
            "<br><b>avg. pred-time (sec)</b> = <b>{:.5f}</b>".format(
                self.id, evaluation_id, avg_pred_time
            )
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        fig.write_html(pr_filepath)

        logger.info('Stored plot under "{}"'.format(pr_filepath))


class NLUResult(object):
    """Natural-Language (Expression-)Understanding Result"""

    def __init__(
        self,
        utterance: str,
        expression_ranking: List["RankingScore"],
        confidence_threshold: float,
        entities: Set["ExtractedEntity"],
    ):
        self.utterance = utterance
        self.expression_ranking = expression_ranking
        self.confidence_threshold = confidence_threshold
        self.entities = set(entities)

    def conf_expression(
        self, confidence_threshold: float = None
    ) -> Optional["RankingScore"]:
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        if len(self.expression_ranking) > 0:
            if self.expression_ranking[0].score >= confidence_threshold:
                return self.expression_ranking[0]
        return None

    def plog(self):
        logger.info("-" * 100)
        logger.info("{}:".format(self.__class__.__name__))

        logger.info(logcolor("expression", "Expression-Ranking:"))
        for e in self.expression_ranking[:15]:
            logger.info(
                logcolor(
                    "expression",
                    "\t{} {}".format(
                        "+" if (e.score >= self.confidence_threshold) else "-", e
                    ),
                )
            )

        logger.info(logcolor("entity", "Entities:"))
        for e in self.entities:
            logger.info(logcolor("entity", "\t● {}".format(e)))
        logger.info("-" * 100)

    def to_repr_dict(self) -> dict:
        return {
            "expression_ranking": [
                e.to_repr_dict() for e in self.expression_ranking[:15]
            ],
            "confidence_threshold": self.confidence_threshold,
            "entities": [e.to_repr_dict() for e in self.entities],
        }


class NLUTrainingData(object):
    def __init__(
        self, intent_filter: List[str], phrase_patterns: List["PhrasePattern"]
    ):
        self.intent_filter = set(intent_filter)
        self.phrase_patterns = set(phrase_patterns)

    def __repr__(self):
        return "({}: #intents={}, #phrase_patterns={})".format(
            self.__class__.__name__, len(self.intent_filter), len(self.phrase_patterns)
        )


class NLUTestData(object):
    def __init__(self, phrase_patterns: List["PhrasePattern"]):
        self.phrase_patterns = set(phrase_patterns)

    def __repr__(self):
        return "({}: #phrase_patterns={})".format(
            self.__class__.__name__, len(self.phrase_patterns)
        )
