import re
import typing
from typing import Optional, List

from dialogue_bot.intent_understanding.iu import (
    ExtractedEntity,
    remove_ambiguous_entities,
)
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import (
    NLUTrainingData,
    create_nlu_train_test_data,
)

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv


class EntityValueExtractor(object):
    """
    Tries to extract Entities out of text
    """

    def __init__(self, env: "BotEnv"):
        self._regex_patterns = set([])  # (entity, value, pattern, needs_context)
        self.env = env

    def init(self, retrain: bool, training_data: Optional["NLUTrainingData"] = None):
        if training_data is None:
            training_data, _ = create_nlu_train_test_data(self.env, test_size=0)

        self._regex_patterns = set([])

        # only use entities that are possible
        entities = self.env.entities(intent_filter=training_data.intent_filter)

        for entity in entities:
            for value in entity.values:

                # add regex-patterns from regex-patterns and synonyms
                for regex_pattern in value.all_regex_patterns:

                    # wrong entity-context
                    if (regex_pattern.entity_context is not None) and (
                        regex_pattern.entity_context != entity.id
                    ):
                        continue

                    self._regex_patterns.add(
                        (
                            entity.id,
                            value.id,
                            "(?P<uifvbzdzg>" + regex_pattern.pattern + ")",
                            regex_pattern.entity_context is not None,
                        )
                    )

        # we know that some of these mappings could be ambiguous, but we will tackle this problem during prediction time

    def run(
        self, entity_context: Optional[str], text: str, intent_filter: List[str] = None
    ) -> List["ExtractedEntity"]:
        """
        :param entity_context: If the entity is already known, put it here, else None
        :param text: The surface text value
        :return:
        """
        # only use entities that are possible
        entity_filter = set(
            [e.id for e in self.env.entities(intent_filter=intent_filter)]
        )
        if entity_context is not None:
            entity_filter.add(entity_context)

        candidates = []
        for (e, v, p, c) in self._regex_patterns:
            if e not in entity_filter:
                continue
            if c and ((entity_context is None) or (entity_context != e)):
                continue

            for m in re.finditer(p, text):
                candidates.append(
                    ExtractedEntity(
                        m.start("uifvbzdzg"),
                        m.end("uifvbzdzg"),
                        e,
                        v,
                        m.group("uifvbzdzg"),
                        1.0,
                        None,
                    )
                )

        # clean up
        candidates = remove_ambiguous_entities(candidates)

        return candidates
