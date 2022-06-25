import logging
import typing
from abc import abstractmethod, ABC
from typing import Set, Optional, List

from dialogue_bot import logcolor
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
    NLUResult,
)
from dialogue_bot.models.input import UserInput
from dialogue_bot.models.utils import KeyComparable

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.models.state import DialogueState
    from dialogue_bot.models.utils import RankingScore

logger = logging.getLogger(__name__)


def entities_overlap(entity1: "ExtractedEntity", entity2: "ExtractedEntity") -> bool:
    if entity2.start <= entity1.start < entity2.end:
        return True
    if entity1.start <= entity2.start < entity1.end:
        return True

    return False


def remove_ambiguous_entities(
    entities: List["ExtractedEntity"],
) -> List["ExtractedEntity"]:
    """
    For overlapping entities with same (entity, value), the longest one will remain.
    """

    remove_idxs = set([])
    for i, entity in enumerate(entities):

        for j, entity2 in enumerate(entities):
            if i == j:
                continue

            if entities_overlap(entity, entity2):
                if (entity.entity == entity2.entity) and (
                    entity.value == entity2.value
                ):
                    # remove the shorter one (or the one with the smallest index, if both have equal length)
                    if len(entity.text) == len(entity2.text):
                        if i <= j:
                            remove_idxs.add(i)
                        else:
                            remove_idxs.add(j)
                    elif len(entity.text) > len(entity2.text):
                        remove_idxs.add(j)
                    else:
                        remove_idxs.add(i)

    return [e for i, e in enumerate(entities) if i not in remove_idxs]


def sort_intent_ranking(
    env: "BotEnv", dialogue_state: "DialogueState", intent_ranking: List["RankingScore"]
) -> List["RankingScore"]:
    # sort primarily by score, next by most recent, next by number of input contexts

    max_context_lived = (
        max([c.lived for c in dialogue_state.contexts])
        if len(dialogue_state.contexts) > 0
        else 0
    )

    def most_recent_input_context_value(max_context_lived, intent) -> int:
        """The smaller the value, the most recent"""
        res = []
        for c in intent.input_contexts:
            if c in dialogue_state.context_names:
                context = dialogue_state.context(c)
                res.append(context.lived)
        return min(res) if len(res) > 0 else (max_context_lived + 1)

    # First, by score, next by most recent, next by number of input contexts
    sort_key = lambda s: (
        s.score,
        -most_recent_input_context_value(max_context_lived, env.intent(s.ref_id)),
        len(env.intent(s.ref_id).input_contexts),
    )
    return sorted(intent_ranking, key=sort_key, reverse=True)


class IU(ABC):
    """
    Intent-Understanding unit (IU).
    Given a user input, this ranks intents and extracts entities.
    """

    @abstractmethod
    def init(self, retrain: bool):
        """
        Should be executed before IU is used. This will either load IU with all of its models (when retrain==False)
        or retrain all of its models (when retrain==True)
        """
        pass

    @abstractmethod
    def run(
        self, user_input: "UserInput", dialogue_state: "DialogueState"
    ) -> "IUResult":
        pass

    @abstractmethod
    def update_entities(
        self, intent_id: str, iu_result: "IUResult", dialogue_state: "DialogueState"
    ) -> Set["ExtractedEntity"]:
        """
        Sometimes, dialogue handling will not use the topmost intent in iu_result, but continue with another intent
         e.g. if it is not confident.
        In this case, this method will update the extracted entities (which were previously extracted based on the
        topmost intent) with the new intent that was chosen.
        """
        pass


class IUResult(object):
    """Intent-Understanding Result"""

    def __init__(
        self,
        user_input: "UserInput",
        intent_ranking: List["RankingScore"],
        confidence_threshold: float,
        entities: Set["ExtractedEntity"],
        nlu_result: Optional["NLUResult"],
        fallback_intent_id: Optional[str],
    ):
        """
        :param user_input:              The user input.
        :param intent_ranking:          The intent ranking
        :param confidence_threshold:    The confidence threshold
        :param entities:                The extracted entities
        :param nlu_result:              The nlu result
        :param fallback_intent_id:      The id of a custom fallback intent that should be executed.
            If None, then the Bot will decide which fallback to execute.
        """

        self.user_input = user_input
        self.intent_ranking = intent_ranking
        self.confidence_threshold = confidence_threshold
        self.entities = entities
        self.nlu_result = nlu_result
        self.fallback_intent_id = fallback_intent_id

    @property
    def conf_intent(self) -> Optional["RankingScore"]:
        if len(self.intent_ranking) > 0:
            if self.intent_ranking[0].score >= self.confidence_threshold:
                return self.intent_ranking[0]
        return None

    def plog(self):
        logger.info("-" * 100)
        logger.info("{}:".format(self.__class__.__name__))

        logger.info(logcolor("intent", "Intent-Ranking:"))
        for e in self.intent_ranking[:15]:
            logger.info(
                logcolor(
                    "intent",
                    "\t{} {}".format(
                        "+" if (e.score >= self.confidence_threshold) else "-", e
                    ),
                )
            )

        logger.info(logcolor("entity", "NL-Entities:"))
        for e in self.entities:
            logger.info(logcolor("entity", "\t● {}".format(e)))

        logger.info(
            logcolor("intent", "Fallback-Intent: {}".format(self.fallback_intent_id))
        )
        logger.info("-" * 100)

    def to_repr_dict(self) -> dict:
        return {
            "intent_ranking": [e.to_repr_dict() for e in self.intent_ranking[:15]],
            "confidence_threshold": self.confidence_threshold,
            "entities": [e.to_repr_dict() for e in self.entities],
            "fallback_intent_id": self.fallback_intent_id,
            "nlu_result": self.nlu_result.to_repr_dict()
            if self.nlu_result is not None
            else None,
        }


class ExtractedEntity(KeyComparable):
    def __init__(
        self,
        start: int,
        end: int,
        entity: str,
        value: Optional[str],
        text: str,
        confidence: float,
        extractor: Optional[str],
    ):
        if end <= start:
            raise ValueError("end cannot be <= start")

        self.start = start
        self.end = end
        self.entity = entity
        self.value = value
        self.text = text
        self.confidence = confidence
        self.extractor = extractor

    def key_tuple(self) -> tuple:
        return (self.entity, self.value, self.text)

    def __repr__(self):
        return '({}: {} "{}" [{}:{}])'.format(
            self.entity, self.value, self.text, self.start, self.end
        )

    def to_repr_dict(self) -> dict:
        return self.__dict__
