import logging
import typing
from typing import List, Union

from dialogue_bot.models.pattern import PhrasePattern
from dialogue_bot.models.utils import KeyComparable

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv

logger = logging.getLogger(__name__)


class NLExpression(KeyComparable):
    """
    A set of phrases and patterns that are used to match user inputs.
    The reason for identifying expressions instead of intents by NLU is that many intents could share same patterns/phrases in their nl-triggers,
    e.g. intents "are_you_ok.yes" and "should_we_continue.yes" would both share phrases such as ["yes", "sure", "yep"].
    Using expressions that can be shared by multiple intents, we get rid of this ambiguity and let NLU only detect one expression instead of multiple intents.
    For our example, intents "are_you_ok.yes" and "should_we_continue.yes" would share the same expression instance within their nl-triggers,
    which would contain phrases ["yes", "sure", "yep"].
    """

    def __init__(
        self,
        env: "BotEnv",
        id: str,
        regex_patterns: List[str] = None,
        exclude_regex_patterns: List[str] = None,
        phrase_patterns: list = None,
    ):

        if regex_patterns is None:
            regex_patterns = []
        if exclude_regex_patterns is None:
            exclude_regex_patterns = []
        if phrase_patterns is None:
            phrase_patterns = []

        self.env = env
        self.id = id
        self.regex_patterns = set(regex_patterns)
        self.exclude_regex_patterns = set(exclude_regex_patterns)
        self.phrase_patterns = set([])
        for x in phrase_patterns:
            self.add_phrase_pattern(x)

    def key_tuple(self) -> tuple:
        return (self.id,)

    def add_phrase_pattern(self, pattern: Union[str, "PhrasePattern"]):
        if isinstance(pattern, str):
            pattern = PhrasePattern(self.id, pattern)
        self.phrase_patterns.add(pattern)

    def __repr__(self):
        phrase_pattern_example = (
            list(self.phrase_patterns)[0].pattern
            if len(self.phrase_patterns) > 0
            else ""
        )
        return '({}: "{}", "{}" [{}])'.format(
            self.__class__.__name__,
            self.id,
            phrase_pattern_example,
            len(self.phrase_patterns),
        )
