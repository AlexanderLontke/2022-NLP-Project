import logging
import re
import typing
from typing import List, Union

from dialogue_bot.models.utils import KeyComparable

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv

logger = logging.getLogger(__name__)


class ValueRegexPattern(KeyComparable):
    """A regex-pattern for a value.
    Example: r'\b0\d\d\d\d\d\d\d\d\d\d\d\b' for value "MOBILE_NUMBER"
    """

    def __init__(self, value: str, pattern: str, entity_context=None):
        """
        Creates a new value-regex-pattern.
        :param value:          The value id
        :param pattern:        The regex pattern
        :param entity_context: An entity-id or None. If provided, this means that this synonym is only valid in the
            given entity-context (e.g. when entity is already known
        """
        self.value = value
        self.pattern = pattern
        self.entity_context = entity_context

    def key_tuple(self) -> tuple:
        return self.value, self.pattern, self.entity_context

    def __repr__(self):
        return '("{}")'.format(self.pattern)


class ValueSynonym(KeyComparable):
    """A synonym for a value.
    Example: 'Germany' for value "DE"
    """

    def __init__(
        self,
        value: str,
        text: str,
        case_sensitive: bool = False,
        entity_context: str = None,
    ):
        """
        Creates a new value-synonym.
        :param value:          The value id
        :param text:           The synonym
        :param case_sensitive: True, if the case should be respected when extracting the value-synonym, else False
        :param entity_context: An entity-id or None. If provided, this means that this synonym is only valid in the
            given entity-context (e.g. when entity is already known)
        """
        self.value = value
        self.text = text
        self.case_sensitive = case_sensitive
        self.entity_context = entity_context

    def to_regex_pattern(self) -> "ValueRegexPattern":
        if self.case_sensitive:
            pattern = r"\b(?:" + re.escape(self.text) + r")\b"
        else:
            pattern = r"\b(?i:" + re.escape(self.text) + r")\b"
        return ValueRegexPattern(self.value, pattern, self.entity_context)

    def key_tuple(self) -> tuple:
        return self.value, self.text, self.case_sensitive, self.entity_context

    def __repr__(self):
        return '("{}")'.format(self.text)


class Value(KeyComparable):
    """
    An value is an instance that occurs in natural language.
    A value can be expressed in different ways in natural language (=> synonyms).
    An examples of a value is "DE" with potential synonyms "Germany", "Deutschland", "BRD", ...
    Here, values will be used as instances for entities (e.g. value "DE" as one of many instances for entity "country").
    However, in this implementation, values co-exist besides entities, meaning that values can also exist without being
    linked by an entity.

    Values should be designed in a way that minimizes their lexical overlap, meaning that having multiple values that
    share the same synonyms or regex_patterns is something that you should avoid since it will cause difficulties to
    decide which of these values is identified in a user utterance.
    """

    def __init__(
        self,
        env: "BotEnv",
        id: str,
        synonyms: List[Union[str, "ValueSynonym"]] = None,
        regex_patterns: List[Union[str, "ValueRegexPattern"]] = None,
    ):
        """
        Creates a new value.
        Synonyms and regex patterns can be used by `NLU` to identify values in a user utterance.
        Therefore, ensure that you provide a substantial amount of synonyms and regex patterns here.
        :param env:            The bot
        :param id:             A unique id for the value
        :param synonyms:       Even if for detection, synonyms are transformed into regex patterns, it is still recommended to
            specify synonyms instead of regex patterns when possible, since they will be used to generate training phrases.
        :param regex_patterns: They should be constructed to work with "re.finditer()".
            So probably, you don't want to use "^" at the beginning or "$" at the and to indicate string boundaries.
        """
        if synonyms is None:
            synonyms = []
            logger.warning(
                'You have not provided synonyms for the entity "{}".'.format(id)
                + "This will not generate any training data if used in NL Intent patterns"
            )
        if regex_patterns is None:
            regex_patterns = []

        self.env = env
        self.id = id
        self.synonyms = set([])
        for x in synonyms:
            self.add_synonym(x)
        self.regex_patterns = set([])
        for x in regex_patterns:
            self.add_regex_pattern(x)

    def key_tuple(self) -> tuple:
        return (self.id,)

    def add_synonym(self, synonym: Union[str, "ValueSynonym"]):
        """Adds a synonym"""
        if isinstance(synonym, str):
            synonym = ValueSynonym(self.id, synonym)
        self.synonyms.add(synonym)

    def add_regex_pattern(self, regex_pattern: Union[str, "ValueRegexPattern"]):
        """Adds a regex pattern"""
        if isinstance(regex_pattern, str):
            regex_pattern = ValueRegexPattern(self.id, regex_pattern)
        self.regex_patterns.add(regex_pattern)

    @property
    def all_regex_patterns(self) -> List["ValueRegexPattern"]:
        """Returns the merged set of regex_patterns and synonym-regex-patterns"""
        res = set([])
        res.update(self.regex_patterns)
        res.update([synonym.to_regex_pattern() for synonym in self.synonyms])
        return list(res)

    def __repr__(self):
        return '({}: "{}", synonyms={})'.format(
            self.__class__.__name__, self.id, [s.text for s in self.synonyms]
        )


if __name__ == "__main__":
    v = Value(None, "TELEPHONE-NR", synonyms=["017635534243"])
    v.add_regex_pattern(r"\b0\d\d\d\d\d\d\d\d\d\d\d\b")
    print(v.all_regex_patterns)
