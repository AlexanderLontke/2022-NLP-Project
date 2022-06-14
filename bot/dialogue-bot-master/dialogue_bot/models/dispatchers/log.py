import logging
import typing
from typing import List

from dialogue_bot import logcolor
from dialogue_bot.models.dispatcher import Dispatcher

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_session import BotSession
    from dialogue_bot.models.action import Action

logger = logging.getLogger(__name__)


class LoggingDispatcher(Dispatcher):
    def after_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        # Log the dialogue state
        bot_session.dialogue_state.plog()

    def utter(self, bot_session: "BotSession", text: str, **kwargs):
        logger.info(logcolor("bot-utterance", '[TEXT] "{}" ({})'.format(text, kwargs)))

    def choice(self, bot_session: "BotSession", choices: List["Choice"], **kwargs):
        logger.info(logcolor("bot-utterance", "[CHOICES] ({})".format(kwargs)))
        for choice in choices:
            logger.info(
                logcolor(
                    "bot-utterance",
                    '\t[{}:{}] "{}" ({})'.format(
                        choice.key, choice.idx, choice.text, choice.args
                    ),
                )
            )

    def media(self, bot_session: "BotSession", media_id: str, **kwargs):
        logger.info(
            logcolor("bot-utterance", '[MEDIA] "{}" ({})'.format(media_id, kwargs))
        )

    def custom(self, bot_session: "BotSession", name: str, **kwargs):
        logger.info(
            logcolor("bot-utterance", '[CUSTOM] "{}" ({})'.format(name, kwargs))
        )
