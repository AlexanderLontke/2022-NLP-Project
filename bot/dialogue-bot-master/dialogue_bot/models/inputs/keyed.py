import copy

from dialogue_bot.models.input import UserInput


class KeyedInput(UserInput):
    """
    A user input with a key and arbitrary parameters
    """

    def __init__(self, key: str, **kwargs):
        self.key = key
        self.args = kwargs

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '({}: "{}", {})'.format(self.__class__.__name__, self.key, self.args)

    def to_repr_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "key": self.key,
            "args": self.args,
        }
