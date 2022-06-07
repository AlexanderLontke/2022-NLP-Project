import logging
import typing

from dialogue_bot.models.action import SlotClearAction
from dialogue_bot.models.inputs.selection import SelectionInput
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.triggers.selection import AnySelectionTrigger
from dialogue_bot.static import actions

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.bot_session import BotSession

logger = logging.getLogger(__name__)


class VerifyIntentIntent(Intent):
    """
    Intent that captures when a user chooses which intent was the one he previously meant.
    """
    def __init__(self, env: 'BotEnv'):
        super().__init__(env, self.__class__.__name__, input_contexts=['verify_intent'], selection_trigger=AnySelectionTrigger())

    def _do_execute(self, session: 'BotSession'):
        input = session.iu_result.user_input
        assert isinstance(input, SelectionInput)

        prev_user_input = session.dialogue_state.slot('verify_intent#input')

        if input.selection_key == 'intent':
            true_intent_id = session.dialogue_state.slot('verify_intent#intents')[input.selection_idx]
            false_intent_ids = [i for i in session.dialogue_state.slot('verify_intent#intents') if i != true_intent_id]

            intent = self.env.intent(true_intent_id)

            # thank user
            actions.utter_verify_intent_thanks(self.env).execute(session)

            # store learned intents
            self.env.db_handler.index_object(self.env.LEARNED_INPUTS_COLLECTION_NAME, {
                'user_input': prev_user_input.to_repr_dict(),
                'true_intent': true_intent_id,
                'false_intents': false_intent_ids
            }, None)
            self.env.db_handler.commit(self.env.LEARNED_INPUTS_COLLECTION_NAME)

            # since we verified the correct intent here, we need to re-extract entities & set entity-slots
            last_iu_result = session.dialogue_state.slot('verify_intent#iu_result')
            last_dialogue_state = session.dialogue_state.slot('verify_intent#dialogue_state')
            updated_entities = self.env.iu.update_entities(true_intent_id, last_iu_result, last_dialogue_state)

            # set entity-slots
            logger.debug('Set Entity Slots.')
            session.update_entity_slots(updated_entities)

            # execute intent action
            intent.action.execute(session)

        elif input.selection_key == 'none':
            false_intent_ids = session.dialogue_state.slot('verify_intent#intents')

            # store (not) learned intents
            self.env.db_handler.index_object(self.env.LEARNED_INPUTS_COLLECTION_NAME, {
                'user_input': prev_user_input.to_repr_dict(),
                'true_intent': None,
                'false_intents': false_intent_ids
            }, None)
            self.env.db_handler.commit(self.env.LEARNED_INPUTS_COLLECTION_NAME)

            actions.utter_verify_intent_none(self.env).execute(session)
        else:
            raise AssertionError('Unknown selection_key: "{}"'.format(input.selection_key))

        # clear slots
        clear_slots = session.dialogue_state.slot('verify_intent#slots')
        for slot in clear_slots:
            SlotClearAction(slot).execute(session)
