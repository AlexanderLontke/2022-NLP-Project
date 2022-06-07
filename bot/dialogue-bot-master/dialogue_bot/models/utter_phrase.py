import re
import typing

from dialogue_bot.intent_understanding.iu import ExtractedEntity
from dialogue_bot.models.utils import islist, KeyComparable
from dialogue_bot.static.enumerations import LANG_ENUMERATIONS

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv


class UtterPhrase(KeyComparable):
    """
    A text that may contain placeholders for slots.
    Examples:
        - "((country)) is a very nice country"
        - "I see that you want to buy a ((car))"
    """
    PLACEHOLDER_PATTERN = r'\(\((?P<placeholder>[^\[\]]*?)\)\)'  # ((country))

    comp_placeholder_pattern = None

    def __init__(self, text):
        self.text = text

        # compile regex patterns
        if UtterPhrase.comp_placeholder_pattern is None:
            UtterPhrase.comp_placeholder_pattern = re.compile(UtterPhrase.PLACEHOLDER_PATTERN)

    def key_tuple(self) -> tuple:
        return self.text,

    def _render_slot(self, env: 'BotEnv', obj) -> str:
        # render list of objects
        if islist(obj):
            return LANG_ENUMERATIONS[env.language](obj, lambda o: self._render_slot(env, o))

        # render an ExtractedEntity
        if isinstance(obj, ExtractedEntity):
            return obj.text

        return str(obj)

    def render_missing(self, slots: dict) -> int:
        res = 0
        for m in UtterPhrase.comp_placeholder_pattern.finditer(self.text):
            k = m.group('placeholder')
            if k not in slots:
                res += 1
        return res

    def render(self, env: 'BotEnv', slots: dict, default_render_func=lambda entity: None) -> str:
        index_offset = 0
        text = self.text
        for m in UtterPhrase.comp_placeholder_pattern.finditer(self.text):
            start, end = m.start() + index_offset, m.end() + index_offset
            k = m.group('placeholder')

            if k in slots:
                replacement = slots[k]
            else:
                replacement = default_render_func(k)

            if replacement is None:
                continue

            replacement_text = self._render_slot(env, replacement)
            text = text[:start] + replacement_text + text[end:]
            index_offset += len(replacement_text) - (end - start)

        return text

    def __repr__(self):
        return '("{}")'.format(self.text)
