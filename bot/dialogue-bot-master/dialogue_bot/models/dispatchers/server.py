import logging
import typing
from typing import List

from dialogue_bot.models.dispatcher import Dispatcher

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_session import BotSession
    from dialogue_bot.models.action import Action

logger = logging.getLogger(__name__)


class ServerDispatcher(Dispatcher):
    def __init__(self):
        self.tracked = {}
        self.responses = []

    def reset(self):
        self.tracked = {}
        self.responses = []

    def track(self, bot_session: "BotSession", **kwargs):
        self.tracked.update(kwargs)

    def after_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        # Log the dialogue state
        bot_session.dialogue_state.plog()

    def _default_dct(self, bot_session: "BotSession", type: str):
        return {
            "type": type,
            "dialogue_state": bot_session.dialogue_state.to_repr_dict(),
        }

    def utter(self, bot_session: "BotSession", text: str, **kwargs):
        TYPE = "text"
        self.responses.append(
            {**self._default_dct(bot_session, TYPE), "text": text, **kwargs}
        )

    def choice(self, bot_session: "BotSession", choices: List["Choice"], **kwargs):
        TYPE = "choice"
        self.responses.append(
            {
                **self._default_dct(bot_session, TYPE),
                "choices": [
                    {"key": c.key, "idx": c.idx, "text": c.text, **c.args}
                    for c in choices
                ],
                **kwargs,
            }
        )

    def media(self, bot_session: "BotSession", media_id: str, **kwargs):
        TYPE = "media"
        self.responses.append(
            {**self._default_dct(bot_session, TYPE), "media_id": media_id, **kwargs}
        )

    def custom(self, bot_session: "BotSession", name: str, **kwargs):
        TYPE = "custom"
        self.responses.append(
            {**self._default_dct(bot_session, TYPE), "name": name, **kwargs}
        )
