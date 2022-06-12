from abc import ABC, abstractmethod
from dialogue_bot.models.inputs.nl import UserInput
from dialogue_bot.models.utter_phrase import UtterPhrase


class ResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, user_input: UserInput) -> str:
        pass
