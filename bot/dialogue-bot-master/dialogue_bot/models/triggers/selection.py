from abc import ABC

from dialogue_bot.models.input import UserInput
from dialogue_bot.models.inputs.selection import SelectionInput
from dialogue_bot.models.trigger import Trigger


class SelectionTrigger(Trigger, ABC):
    pass


class SingleSelectionTrigger(SelectionTrigger):
    """ Triggered, when the user selects a specific option of a selection """

    def __init__(self, selection_key: str, selection_idx: int):
        """
        :param selection_key: The option's selection-key
        :param selection_idx: The option's selection-index
        """
        self.selection_key = selection_key
        self.selection_idx = int(selection_idx)

    def matches_input(self, input: 'UserInput') -> bool:
        return isinstance(input, SelectionInput) and (input.selection_idx == self.selection_idx) and (input.selection_key == self.selection_key)

    def __repr__(self):
        return '({}: "{}":{})'.format(self.__class__.__name__, self.selection_key, self.selection_idx)


class AnySelectionTrigger(SelectionTrigger):
    """ Triggered, when the user selects any option of a selection """

    def matches_input(self, input: 'UserInput') -> bool:
        return isinstance(input, SelectionInput)

    def __repr__(self):
        return '({})'.format(self.__class__.__name__)

