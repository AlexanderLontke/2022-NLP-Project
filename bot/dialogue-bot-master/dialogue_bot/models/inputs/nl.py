import copy

from dialogue_bot.models.input import UserInput


class NLInput(UserInput):
    """
    A user input in natural language
    """

    MIN_LENGTH = 0
    MAX_LENGTH = 400

    def __init__(self, text: str):
        self.text = text.strip()

    def ignore(self) -> "bool":
        return len(self.text) <= 0

    def is_valid(self) -> "bool":
        if not (self.MIN_LENGTH <= len(self.text) <= self.MAX_LENGTH):
            return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '({}: "{}")'.format(self.__class__.__name__, self.text)

    def to_repr_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "text": self.text,
        }
