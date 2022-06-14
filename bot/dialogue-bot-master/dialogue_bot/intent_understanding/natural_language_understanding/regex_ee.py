import logging
import typing
from typing import List, Optional

from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
    NLU,
    NLUResult,
)
from dialogue_bot.intent_understanding.natural_language_understanding.tools.entity_value_extractor import (
    EntityValueExtractor,
)

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
        NLUTrainingData,
    )

logger = logging.getLogger(__name__)


class RegexEntityExtractor(NLU):
    """
    This NLU only extracts entities => no expression ranking.
    This NLU uses regex-patters (include & exclude regex patterns from `NLExpression`) to identify Expressions.
    If a match was found, the expression gets a score of 1 while all others get 0.
    """

    def __init__(self, env: "BotEnv", id: str):
        super().__init__(env, id)
        self._entity_value_extractor = EntityValueExtractor(env)

    def init(self, retrain: bool, training_data: Optional["NLUTrainingData"] = None):
        logger.info(
            "Initializing {} (retrain={}) ...".format(self.__class__.__name__, retrain)
        )

        self._entity_value_extractor.init(retrain, training_data=training_data)

    def run(self, utterance: str, intent_filter: List[str] = None) -> "NLUResult":
        entities = self._entity_value_extractor.run(
            None, utterance, intent_filter=intent_filter
        )
        return NLUResult(utterance, [], 1.0, set(entities))
