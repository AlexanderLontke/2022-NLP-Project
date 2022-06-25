import typing
from typing import List

from dialogue_bot.models.utils import KeyComparable
from dialogue_bot.static.texts import LANG_TEXTS

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.models.value import Value


class Entity(KeyComparable):
    """
    An entity is a concept or class that occurs in natural language.
    An entity can have many instances (here: values).
    An examples of an entity is "country" with potential values "DE", "CH", "FR", ...
    In the context of a chatbot, we want to extract Entities with their values from user utterances in natural language.
    """

    def __init__(
        self,
        env: "BotEnv",
        id: str,
        name: str = None,
        value_refs: List[str] = None,
        default_values: List[str] = None,
        questions: List[str] = None,
    ):
        """
        Creates an entity.
        :param env:             The bot
        :param id:              A unique id for the entity
        :param name:            human readable name
        :param value_refs:      the possible values for that entity, specified by their ids.
        :param default_values:  the default values for that entity, specified by their ids.
        :param questions:       questions that should be asked if the user is asked to provide a value for that entity
            (e.g. ["What is your car?", "In which car are you interested in?")
        """
        if name is None:
            name = id
        if value_refs is None:
            value_refs = []
        if default_values is None:
            default_values = []
        if questions is None:
            questions = []
        if len(questions) <= 0:
            questions = LANG_TEXTS["default_entity_questions"][env.language](name)

        self.env = env
        self.name = name
        self.id = id
        self.value_refs = set(value_refs)
        self.default_values = set(default_values)
        self.questions = set(questions)

    def key_tuple(self) -> tuple:
        return (self.id,)

    @property
    def values(self) -> List["Value"]:
        return [self.env.value(value_ref) for value_ref in self.value_refs]

    def __repr__(self):
        return '({}: "{}", {})'.format(
            self.__class__.__name__, self.id, self.value_refs
        )
