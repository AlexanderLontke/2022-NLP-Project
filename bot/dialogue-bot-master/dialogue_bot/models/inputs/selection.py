import copy

from dialogue_bot.models.input import UserInput


class SelectionInput(UserInput):
    """
    An input where the user selected a specific option of a selection
    (specified by selection_key and selection_idx)
    """
    def __init__(self, selection_key: str, selection_idx: int):
        self.selection_key = selection_key
        self.selection_idx = int(selection_idx)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '({}: "{}":{})'.format(self.__class__.__name__, self.selection_key, self.selection_idx)

    def to_repr_dict(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'selection_key': self.selection_key,
            'selection_idx': self.selection_idx,
        }

