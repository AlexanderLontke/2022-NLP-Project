import logging
import typing
from abc import ABC

from dialogue_bot.models.input import UserInput
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.trigger import Trigger

if typing.TYPE_CHECKING:
    from dialogue_bot.models.expression import NLExpression

logger = logging.getLogger(__name__)


class NLTrigger(Trigger, ABC):
    pass


class PhraseNLTrigger(NLTrigger):
    """Triggered, when there is an utterance in natural language that matches some predefined phrases/patterns"""

    def __init__(self, expression: "NLExpression"):
        self.expression = expression

    def matches_input(self, input: "UserInput") -> bool:
        return isinstance(input, NLInput)

    def __repr__(self):
        return "({}: {})".format(self.__class__.__name__, self.expression)


class AnyNLTrigger(NLTrigger):
    """Triggered, when there is an arbitrary utterance in natural language"""

    def matches_input(self, input: "UserInput") -> bool:
        return isinstance(input, NLInput)

    def __repr__(self):
        return "({})".format(self.__class__.__name__)


class FallbackNLTrigger(NLTrigger):
    """Triggered, when there is an utterance in natural language and no intent could be determined"""

    def matches_input(self, input: "UserInput") -> bool:
        return isinstance(input, NLInput)

    def __repr__(self):
        return "({})".format(self.__class__.__name__)
