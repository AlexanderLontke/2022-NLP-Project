import re
import typing
from typing import Optional, Iterable

from dialogue_bot.models.phrase import Phrase, PhraseEntity
from dialogue_bot.models.utils import KeyComparable
from dialogue_bot.utils.text import replace_spans

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv


class PhrasePattern(KeyComparable):
    """
    A phrase-pattern is a string-representation of a phrase that is annotated with entities/values.
    A phrase-pattern is generally used to generate training phrases for NLU.

    A phrase-pattern can be labeled with entities/values using this pattern:
        - "[text][value_id](entity_id)" such as "[Germany][DE][country]
             Where entity_id refers to an existing entity_id.
             Where value_id refers to an existing value_id for entity_id.
             Where text is the surface-pattern that is labelled with the entity/value information.

    Phrase-patterns can also include placeholder-patterns such as:
        - "((entity_id))" such as "((country))
            Where entity_id refers to an existing entity_id.
            When expanded, this generates (training) phrases with annotated value-instances for that entity,
            e.g. "[Germany][DE](country)" or "[Switzerland][CH](country)"
    """

    # "Hello"
    # "I live in ((country))"
    # "I live in [Germany][DE](country)"
    ENTITY_PLACEHOLDER_PATTERN = r"\(\((?P<entity>[^\[\]]*?)\)\)"
    ENTITY_ANNOTATION_PATTERN = r"(?:\[(?P<surface>[^\[\]]*?)\])(?:\[(?P<value>[^\[\]]*?)\])(?:\((?P<entity>[^\(\)]*?)\))"

    comp_entity_placeholder_pattern = None
    comp_entity_annotation_pattern = None

    def __init__(self, expression_id: str, pattern: str):
        self.expression_id = expression_id
        self.pattern = pattern

        # compile regex patterns
        if PhrasePattern.comp_entity_placeholder_pattern is None:
            PhrasePattern.comp_entity_placeholder_pattern = re.compile(
                PhrasePattern.ENTITY_PLACEHOLDER_PATTERN
            )
        if PhrasePattern.comp_entity_annotation_pattern is None:
            PhrasePattern.comp_entity_annotation_pattern = re.compile(
                PhrasePattern.ENTITY_ANNOTATION_PATTERN
            )

    def key_tuple(self) -> tuple:
        return (self.expression_id, self.pattern)

    def _expand_entity_placeholders(
        self,
        env: "BotEnv",
        max_phrases: Optional[int],
        max_entity_values: int = None,
        max_value_synonyms: int = None,
    ) -> Iterable[str]:
        """
        Replaces "((entity))" placeholders in the text with real entity-annotations "[text][value](entity)".
        E.g.:
            "I live in ((country))":
                - "I live in [Germany][DE](country)"
                - "I live in [Deutschland][DE](country)"
                - "I live in [BRD][DE](country)"
                - "I live in [Switzerland][CH](country)"
                - "I live in [Schweiz][CH](country)"
                - ...

        :param max_phrases: The maximum amount of generated phrases. If None, then all phrase possibilities are generated.
        :param max_entity_values: The maximum amount of values that an entity is replaced with during the generation of phrases.
        :param max_value_synonyms: The maximum amount of synonyms that an entity-value is replaced with during the generation of phrases.
        """

        # create replacements
        replacements = []
        for m in PhrasePattern.comp_entity_placeholder_pattern.finditer(self.pattern):
            entity_id = m.group("entity")
            entity = env.entity(entity_id)

            repl_values = []

            nr_entity_values = 0  # how often the entity was replaced with a value
            for value in entity.values:
                if (max_entity_values is not None) and (
                    max_entity_values <= nr_entity_values
                ):
                    break

                nr_value_synonyms = 0  # how often the value was replaced with a synonym
                for synonym in value.synonyms:
                    if (max_value_synonyms is not None) and (
                        max_value_synonyms <= nr_value_synonyms
                    ):
                        break

                    repl_values.append(
                        "[{}][{}]({})".format(synonym.text, synonym.value, entity.id)
                    )
                    nr_value_synonyms += 1

                nr_entity_values += 1

            replacements.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "values": repl_values,
                    "stored__entity": entity_id,
                }
            )

        # replace
        for pattern in replace_spans(
            self.pattern, replacements, max_realizations=max_phrases
        ):
            yield pattern

    def _phrase_from_pattern(self, pattern: str) -> "Phrase":
        """
        "I live in [Germany][DE](country)" =>  ("I live in Germany" | (country: DE "Germany"))
        """

        # create replacements
        replacements = []
        for m in PhrasePattern.comp_entity_annotation_pattern.finditer(pattern):
            surface = m.group("surface")
            value = m.group("value")
            entity_id = m.group("entity")
            repl_val = {"value": value, "surface": surface}

            replacements.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "values": [repl_val],
                    "stored__entity": entity_id,
                }
            )

        # replace
        text, replacements = replace_spans(
            pattern,
            replacements,
            value_func=lambda t: t["surface"],
            output_replacements=True,
        )[0]
        entities = [
            PhraseEntity(
                repl["res_start"],
                repl["res_end"],
                repl["stored__entity"],
                repl["res_value"]["value"],
            )
            for repl in replacements
        ]

        return Phrase(self.expression_id, text, entities)

    def generate_phrases(
        self,
        env: "BotEnv",
        max_phrases: Optional[int],
        max_entity_values: int = None,
        max_value_synonyms: int = None,
    ) -> Iterable["Phrase"]:
        for pattern in self._expand_entity_placeholders(
            env,
            max_phrases,
            max_entity_values=max_entity_values,
            max_value_synonyms=max_value_synonyms,
        ):
            yield self._phrase_from_pattern(pattern)

    def __repr__(self):
        return '("{}")'.format(self.pattern)


if __name__ == "__main__":
    from dialogue_bot.bot_env import BotEnv

    env = BotEnv("test", "en")
    # env._init_dbs(True)
    # env._init_intents()
    #
    # env.register_entity('country')
    # env.register_entity_values('country', ['DE', 'CH', 'IT'])
    # env.register_synonyms('DE', ['Germany', 'Deutschland'])
    # env.register_synonyms('CH', ['Switzerland', 'Schweiz'])
    # env.register_synonyms('IT', ['Italy', 'Italien'])
    #
    # print()
    # p = PhrasePattern(None, 'I live in [Germany][DE](country) and work in ((country))')
    # for x in p.generate_phrases(env, None):
    #     print(x)
    # print()
    # for x in p.generate_phrases(env, None, one_for_each_value=True):
    #     print(x)
    #
    # print()
    # p = PhrasePattern(None, 'Hello')
    # for x in p.generate_phrases(env, None):
    #     print(x)
