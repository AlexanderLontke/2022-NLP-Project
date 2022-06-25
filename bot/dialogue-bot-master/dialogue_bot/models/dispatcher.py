import logging
import typing
from abc import abstractmethod, ABC
from typing import List

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_session import BotSession
    from dialogue_bot.models.action import Action

logger = logging.getLogger(__name__)


class Dispatcher(ABC):
    """A class used to deliver chatbot actions"""

    def track(self, bot_session: "BotSession", **kwargs):
        """Called anytime the track() method is called"""
        pass

    def before_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        """Called before an action is executed"""
        pass

    def after_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        """Called after an action was executed"""
        pass

    def end(self, bot_session: "BotSession", **kwargs):
        """Called when the chatbot is finished in producing responses"""
        pass

    @abstractmethod
    def utter(self, bot_session: "BotSession", text: str, **kwargs):
        """Called when the chatbot utters a message in natural language"""
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def choice(
        self,
        bot_session: "BotSession",
        choices: List["Choice"],
        text: str = None,
        **kwargs
    ):
        """Called when the chatbot gives a choice"""
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def media(
        self,
        bot_session: "BotSession",
        media_id: str,
        media_type: str = None,
        title: str = None,
        desc: str = None,
        **kwargs
    ):
        """Called when the chatbot shows a media object"""
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def custom(self, bot_session: "BotSession", name: str, **kwargs):
        """Called when the chatbot should return a custom object"""
        raise NotImplementedError("Implement in subclass")


class MultiDispatcher(Dispatcher):
    def __init__(self, dispatchers):
        self.dispatchers = dispatchers

    def track(self, bot_session: "BotSession", **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.track(bot_session, **kwargs)

    def before_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.before_action(bot_session, action, **kwargs)

    def after_action(self, bot_session: "BotSession", action: "Action", **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.after_action(bot_session, action, **kwargs)

    def end(self, bot_session: "BotSession", **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.end(bot_session, **kwargs)

    def utter(self, bot_session: "BotSession", text: str, **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.utter(bot_session, text, **kwargs)

    def choice(
        self,
        bot_session: "BotSession",
        choices: List["Choice"],
        text: str = None,
        **kwargs
    ):
        for dispatcher in self.dispatchers:
            dispatcher.choice(bot_session, choices, text=text, **kwargs)

    def media(
        self,
        bot_session: "BotSession",
        media_id: str,
        media_type: str = None,
        title: str = None,
        desc: str = None,
        **kwargs
    ):
        for dispatcher in self.dispatchers:
            dispatcher.media(
                bot_session,
                media_id,
                media_type=media_type,
                title=title,
                desc=desc,
                **kwargs
            )

    def custom(self, bot_session: "BotSession", name: str, **kwargs):
        for dispatcher in self.dispatchers:
            dispatcher.custom(bot_session, name, **kwargs)


class AccumulateDispatcher(Dispatcher):
    def __init__(self):
        self.responses = []

    def utter(self, bot_session: "BotSession", text: str, **kwargs):
        self.responses.append({"type": "utter", "message": text})

    def choice(
        self,
        bot_session: "BotSession",
        choices: List["Choice"],
        text: str = None,
        **kwargs
    ):
        self.responses.append(
            {
                "type": "choice",
                "choices": [
                    {"key": c.key, "idx": c.idx, "text": c.text, **c.args}
                    for c in choices
                ],
            }
        )

    def media(
        self,
        bot_session: "BotSession",
        media_id: str,
        media_type: str = None,
        title: str = None,
        desc: str = None,
        **kwargs
    ):
        self.responses.append(
            {
                "type": "media",
                "media_id": media_id,
            }
        )

    def custom(self, bot_session: "BotSession", name: str, **kwargs):
        self.responses.append(
            {
                "type": "custom",
                "name": name,
            }
        )
