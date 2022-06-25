import logging
from abc import abstractmethod, ABC

from dialogue_bot.models.input import UserInput

logger = logging.getLogger(__name__)


class Trigger(ABC):
    """A Trigger defines how an intent can be triggered (e.g. through natural language, selection clicks, ...)"""

    @abstractmethod
    def matches_input(self, input: "UserInput") -> bool:
        """Determines if the trigger can process the input"""
        pass
