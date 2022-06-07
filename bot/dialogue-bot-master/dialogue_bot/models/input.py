from abc import ABC, abstractmethod


class UserInput(ABC):
    """ A user input """

    def ignore(self) -> 'bool':
        """ Returns if the input should be ignored by the bot. In this case, the bot will not progress or respond
        to the message"""
        return False

    def is_valid(self) -> 'bool':
        """ Returns if the input is invalid. In this case, the bot will respond with a error response."""
        return True

    @abstractmethod
    def to_repr_dict(self) -> dict:
        return {
            'type': self.__class__.__name__
        }
