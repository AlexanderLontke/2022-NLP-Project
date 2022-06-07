import logging
import typing
from typing import Iterable

from dialogue_bot.models.dispatchers.log import LoggingDispatcher
from dialogue_bot.models.state import DialogueState

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.models.dispatcher import Dispatcher
    from dialogue_bot.intent_understanding.iu import ExtractedEntity, IUResult

logger = logging.getLogger(__name__)


class BotSession(object):
    """
    This class stores the dialogue state for a specific session with the chatbot.
    Sessions can be used to allow multiple conversations with the chatbot in parallel.
    By providing multiple users with their own sessions, their individual conversations with the bot do not interfere.
    """

    def __init__(self, env: 'BotEnv', dialogue_state: 'DialogueState' = None, dispatcher: 'Dispatcher' = None):
        """
        Creates a session.
        :param env:            The bot.
        :param dialogue_state: [Optional] A dialogue state that you want to start with, else creates a new one
        :param dispatcher:     [Optional] The dispatcher that you want to use, else creates a new `LoggingDispatcher`
        """
        if dialogue_state is None:
            dialogue_state = env.create_dialogue_state()
        if dispatcher is None:
            dispatcher = LoggingDispatcher()

        self.env = env
        self.dialogue_state = dialogue_state
        self.dispatcher = dispatcher
        self.iu_result: 'IUResult' = None

    def reset_dialogue(self):
        """ Resets the dialogue back to the start """
        self.dialogue_state = self.env.create_dialogue_state()

    def update_entity_slots(self, entities: Iterable['ExtractedEntity']):
        """
        Sets the default entity values and the passed entities.
        The passed entities will override the default values in this process.
        """

        # set default entity-values
        for entity in self.env.entities():
            if entity.default_values is not None:
                self.dialogue_state.set_slot(entity.id, entity.default_values)

        #  entity-values
        nl_slots = {}
        for ex_entity in entities:
            nl_slots[ex_entity.entity] = nl_slots.get(ex_entity.entity, set([]))
            nl_slots[ex_entity.entity].add(ex_entity)
        for k, vs in nl_slots.items():
            self.dialogue_state.set_slot(k, vs)



