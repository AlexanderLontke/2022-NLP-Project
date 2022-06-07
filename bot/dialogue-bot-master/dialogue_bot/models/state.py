import copy
import logging
from typing import List, Dict, Optional

import jsonpickle
from dialogue_bot import logcolor
from dialogue_bot.models.context import TTLContext

logger = logging.getLogger(__name__)


class DialogueState(object):
    """
    The place where all dialogue progress is being stored.

        - Contexts are used to store dialogue states or dialogue milestones.
            Each intent has a set of input contexts that need to be available in the DialogueState so that the input can be matched.
            Therefore, contexts can be used to control which intents are active.
            One use case are followup intents, where the first intent would set a context (e.g. "greet") in the dialogue state and the follow-up intent uses
            "greet" as an input-context. Therefore, the follow-up intent is only active after the first intent was executed.
            A context typically has a lifetime that decreases with each user interaction. Once the lifetime is over, the context is deleted from the dialogue state.

        - Slots are used to store information as key-value pairs during the dialogue.
            They can be accessed and modified by each intent.
            Therefore, slots can be seen as the "memory" of the chatbot.
    """

    def __init__(self,
                 contexts: List['TTLContext'] = None,
                 slots: Dict[str, any] = None,
                 ):
        if contexts is None:
            contexts = []
        if slots is None:
            slots = {}

        self._contexts = {}
        for context in contexts:
            self.set_context(context)

        self._slots = {}
        for k, v in slots.items():
            self.set_slot(k, v)

    # CONTEXTS #########################################################################################################

    def set_context(self, context):
        if not isinstance(context, TTLContext):
            context = TTLContext(context)
        self._contexts[context.name] = copy.deepcopy(context)

    def clear_context(self, name):
        if name in self._contexts:
            del self._contexts[name]

    def context(self, name) -> Optional['TTLContext']:
        return self._contexts.get(name, None)

    @property
    def contexts(self) -> List['TTLContext']:
        return list(self._contexts.values())

    @property
    def context_names(self) -> List[str]:
        return [c.name for c in self._contexts.values()]

    # SLOTS ############################################################################################################

    def set_slot(self, slot: str, value):
        """ Note that value can also be a list of values """
        self._slots[slot] = value

    def clear_slot(self, slot: str):
        if slot in self._slots:
            del self._slots[slot]

    def slot(self, slot: str, default=None):
        return self._slots.get(slot, default)

    @property
    def slots(self) -> Dict[str, str]:
        return self._slots

    # OTHER ############################################################################################################

    def live(self):
        for context in self._contexts.values():
            context.live()
        self._contexts = {n: c for n, c in self._contexts.items() if not c.is_dead}

    def plog(self):
        logger.info('-' * 100)
        logger.debug('Current Dialogue State:')
        logger.debug(logcolor('context', '\tContexts:'))
        for c in sorted(self.contexts, key=lambda x: (x.remaining if x.remaining is not None else 0)):
            logger.info(logcolor('context', '\t\t● {}: {} {}'.format(c.name, c.remaining, c.lived)))
        logger.debug(logcolor('slot', '\tSlots:'))
        for k, v in self.slots.items():
            logger.info(logcolor('slot', '\t\t● {}: {}'.format(k, v)))
        logger.info('-' * 100)

    def to_repr_dict(self) -> dict:
        return {
            'contexts': [{'name': c.name, 'remaining': c.remaining, 'lived': c.lived} for c in self.contexts],
            'slots': [{'slot': k, 'value': str(v)} for k, v in self.slots.items()]
        }

    def to_json(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(s):
        return jsonpickle.decode(s)

    def clone(self):
        return self.from_json(self.to_json())

    def __repr__(self):
        return '({}: contexts={}, slots={})'.format(self.__class__.__name__, self.contexts, self.slots)
