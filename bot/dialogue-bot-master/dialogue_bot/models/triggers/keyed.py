import logging
import typing

from dialogue_bot.models.input import UserInput
from dialogue_bot.models.inputs.keyed import KeyedInput
from dialogue_bot.models.trigger import Trigger

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class KeyedTrigger(Trigger):
    """ Triggered by a `KeyedInput` that has the specified key """

    def __init__(self, key: str):
        self.key = key

    def matches_input(self, input: 'UserInput') -> bool:
        return isinstance(input, KeyedInput) and (input.key == self.key)

    def __repr__(self):
        return '({}: "{}")'.format(self.__class__.__name__, self.key)