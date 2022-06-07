import datetime
import logging
import os
import threading
from typing import Union, List, Dict

from dialogue_bot import logcolor, project_settings
from dialogue_bot.bot_session import BotSession
from dialogue_bot.database.db_handler import DBHandler, IndexHandler
from dialogue_bot.intent_understanding.default_iu import DefaultIU
from dialogue_bot.models.context import TTLContext
from dialogue_bot.models.dispatcher import MultiDispatcher, AccumulateDispatcher
from dialogue_bot.models.entity import Entity
from dialogue_bot.models.expression import NLExpression
from dialogue_bot.models.input import UserInput
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.pattern import PhrasePattern
from dialogue_bot.models.state import DialogueState
from dialogue_bot.models.triggers.nl import PhraseNLTrigger
from dialogue_bot.models.utils import islist
from dialogue_bot.models.value import Value
from dialogue_bot.static import actions
from dialogue_bot.static.actions import verify_intent
from dialogue_bot.static.intents import VerifyIntentIntent
from dialogue_bot.utils.score_clustering import top_similar_scores

logger = logging.getLogger(__name__)


class BotEnv(object):
    """
    The main Bot class.
    Keeps everything that is needed to respond to a user input for a given `BotSession`.
    For designing the chatbot dialogue, you need to register Intents, Entities and Values using the register-functions.
    After registering all of these, you need to call the `start()` method in order to use the chatbot.
    After that, the chatbot should be able to respond to user inputs for specific `BotSession` instances (see `respond()`).
    """
    LOGGING_COLLECTION_NAME = 'logging'
    LEARNED_INPUTS_COLLECTION_NAME = 'learned-inputs'

    def __init__(self, id: str, language: str,
                 max_verify_nr=15,
                 start_contexts: List['TTLContext'] = None,
                 start_slots: Dict[str, any] = None,
                 log_conversation: bool = True
                 ):
        """
        Creates a bot.
        :param id:                  A unique id for the chatbot
        :param language:            The language for the chatbot (Currently supported: `en`)
        :param max_verify_nr:       [Optional] If the chatbot did not understand the utterance, it will provide the user with
            a list of possible candidates. This describes the maximum number for these candidates that are shown to the user.
        :param start_contexts:      [Optional] The contexts that are active at the start of any new conversation with the chatbot.
        :param start_slots:         [Optional] The slots that are set at the start of any new conversation with the chatbot.
        :param log_conversation     If the bot should log & store the conversation into into a database
        """
        self.id = id
        self.language = language
        self.max_verify_nr = max_verify_nr
        self.start_contexts = start_contexts
        self.start_slots = start_slots
        self.log_conversation = log_conversation

        self._intents = {}
        self._entities = {}
        self._values = {}

        self.iu = DefaultIU(self)
        self.db_handler = DBHandler(self.id)
        self.index_handler = IndexHandler(os.path.join(self.storage_dir, 'index'))

        # ensure some collections
        self.db_handler.ensure_collection(self.LOGGING_COLLECTION_NAME)
        self.db_handler.ensure_collection(self.LEARNED_INPUTS_COLLECTION_NAME)

    @property
    def storage_dir(self) -> str:
        """ The path where chatbot data is stored """
        return os.path.join(project_settings.GENERATED_DATA_DIRPATH, self.id)

    def create_dialogue_state(self) -> 'DialogueState':
        """ Creates a new DialogueState for a new conversation with the chatbot """
        return DialogueState(contexts=self.start_contexts, slots=self.start_slots)

    def ensure_nl_trigger(self, intent: 'Intent'):
        """ Ensures that a NLTrigger for the intent exists """
        if (intent.nl_trigger is None) or (not isinstance(intent.nl_trigger, PhraseNLTrigger)):
            intent.nl_trigger = PhraseNLTrigger(NLExpression(self, intent.id + '-expression'))

    # REGISTER-METHODS #################################################################################################

    def register_intent(self, intent: Union[str, 'Intent'], **kwargs):
        """
        Registers an intent either by passing an `Intent`-instance
        or a (new) intent id with its constructor arguments as kwargs.
        """
        if isinstance(intent, str):
            intent = Intent(self, intent, **kwargs)

        self._intents[intent.id] = intent
        logger.info(logcolor('intent', 'Registered {}'.format(intent)))

    def register_entity(self, entity: Union[str, 'Entity'], **kwargs):
        """
        Registers an entity either by passing an `Entity`-instance
        or a (new) entity id with its constructor arguments as kwargs.
        """
        if isinstance(entity, str):
            entity = Entity(self, entity, **kwargs)

        self._entities[entity.id] = entity
        logger.info(logcolor('entity', 'Registered {}'.format(entity)))

    def register_value(self, value: Union[str, 'Value'], **kwargs):
        """
        Registers a value either by passing an `Value`-instance
        or a (new) value id with its constructor arguments as kwargs.
        """
        if isinstance(value, str):
            value = Value(self, value, **kwargs)

        self._values[value.id] = value
        logger.info(logcolor('value', 'Registered {}'.format(value)))

    # EXT REGISTER-METHODS #############################################################################################
    # these are just additional register-methods for improved usability

    def register_intent_phrase_patterns(self, intent_id: str, phrase_pattern: Union[list, str, 'PhrasePattern']):
        """
        Registers one or many phrase patterns for an intent. (Intent needs to be registered first)
        :param intent_id: The intent-id.
        :param phrase_pattern: string-pattern or instance of `PhrasePattern` or a list of those.
        """
        if islist(phrase_pattern):
            for x in phrase_pattern:
                self.register_intent_phrase_patterns(intent_id, x)
            return

        intent = self.intent(intent_id)
        self.ensure_nl_trigger(intent)
        intent.nl_trigger.expression.add_phrase_pattern(phrase_pattern)

    def register_intent_regex_patterns(self, intent_id: str, regex_patterns: Union[list, str]):
        """
        Registers one or many regex patterns for an intent. (Intent needs to be registered first)
        :param intent_id: The intent-id.
        :param phrase_pattern: string-regex-pattern or a list of string-regex-patterns.
        """
        if islist(regex_patterns):
            for x in regex_patterns:
                self.register_intent_regex_patterns(intent_id, x)
            return

        intent = self.intent(intent_id)
        self.ensure_nl_trigger(intent)
        intent.nl_trigger.expression.regex_patterns.add(regex_patterns)

    def register_intent_exclude_regex_patterns(self, intent_id: str, regex_patterns: Union[list, str]):
        """
        Registers one or many exclude regex patterns for an intent. (Intent needs to be registered first)
        :param intent_id: The intent-id.
        :param phrase_pattern: string-regex-pattern or a list of string-regex-patterns.
        """
        if islist(regex_patterns):
            for x in regex_patterns:
                self.register_intent_exclude_regex_patterns(intent_id, x)
            return

        intent = self.intent(intent_id)
        self.ensure_nl_trigger(intent)
        intent.nl_trigger.expression.exclude_regex_patterns.add(regex_patterns)

    # BOT-LIFECYCLE ####################################################################################################

    def start(self, reset: bool):
        """
        Needs to be called when registration is done and before the chatbot is used.
        :param reset: True, if the chatbot should be reset. This includes retraining all of its models
        """
        logger.info('########## START BOT ##########')

        # add intents
        self.register_intent(VerifyIntentIntent(self))

        # init iu
        logger.info('Init IU')
        self.iu.init(reset)

        logger.info('########## BOT READY ##########')

    # ACCESSORS ########################################################################################################

    def intents(self,
                intent_filter: List[str] = None,
                domain_filter: List[str] = None,
                nl_expression_filter: List[str] = None) -> List['Intent']:
        """
        Returns a list of intents.
        :param intent_filter: A list of intent-ids
        :param domain_filter: Every intent should have at least belong to one of those domains
        :param nl_expression_filter: Return only intents that have one of these expressions
        """
        res = []
        for intent in self._intents.values():
            if (intent_filter is not None) and (intent.id not in intent_filter):
                continue
            if (domain_filter is not None) and all([(d not in intent.domains) for d in domain_filter]):
                continue
            if (nl_expression_filter is not None) and (
                    (intent.nl_trigger is None) or (not isinstance(intent.nl_trigger, PhraseNLTrigger)) or (
                    intent.nl_trigger.expression.id not in nl_expression_filter)):
                continue

            res.append(intent)

        return res

    def intent(self, intent_id: str) -> 'Intent':
        """ Returns a specific intent by its id. """
        if intent_id not in self._intents:
            raise ValueError(
                'Intent "{}" is not registered yes. Please register first!'.format(logcolor('intent', intent_id)))

        return self._intents[intent_id]

    def entities(self,
                 entity_filter: List[str] = None,
                 intent_filter: List[str] = None,
                 ) -> List['Entity']:
        """
        Returns a list of entities.
        :param entity_filter: A list of entity-ids
        :param intent_filter: Only return entities that are used by those intents
        """
        res = []
        for entity in self._entities.values():

            if (entity_filter is not None) and (entity.id not in entity_filter):
                continue

            if intent_filter is not None:
                entity_used = False
                for intent in self.intents(intent_filter=intent_filter):
                    if (intent.entity_filter is None) or (entity.id in intent.entity_filter):
                        entity_used = True
                        break
                if not entity_used:
                    continue

            res.append(entity)
        return res

    def entity(self, entity_id: str) -> 'Entity':
        """ Returns a specific entity by its id. """
        if entity_id not in self._entities:
            raise ValueError(
                'Entity "{}" is not registered yes. Please register first!'.format(logcolor('entity', entity_id)))

        return self._entities[entity_id]

    def values(self,
               value_filter: List[str] = None,
               entity_filter: List[str] = None) -> List['Value']:
        """
        Returns a list of values.
        :param value_filter: A list of value-ids
        :param entity_filter: Only return values that are used by those entities
        """
        res = []
        for value in self._values.values():

            if (value_filter is not None) and (value.id not in value_filter):
                continue

            if entity_filter is not None:
                value_used = False
                for entity in self.entities(entity_filter=entity_filter):
                    if value.id in entity.value_refs:
                        value_used = True
                        break
                if not value_used:
                    continue

            res.append(value)
        return res

    def value(self, value_id: str) -> 'Value':
        """ Returns a specific value by its id. """
        if value_id not in self._values:
            raise ValueError(
                'Value "{}" is not registered yes. Please register first!'.format(logcolor('value', value_id)))

        return self._values[value_id]

    # EXT ACCESSORS ####################################################################################################

    def nl_expressions(self, intent_filter: List[str] = None) -> List['NLExpression']:
        """
        Returns a list of NL-expressions.
        :param intent_filter: Only return expressions that are used by those intents
        """
        res = set()
        for intent in self.intents(intent_filter=intent_filter):
            if (intent.nl_trigger is None) or (not isinstance(intent.nl_trigger, PhraseNLTrigger)):
                continue
            res.add(intent.nl_trigger.expression)
        return list(res)

    # DIALOGUE HANDLING ################################################################################################

    def reset_dialogue(self, bot_session: 'BotSession'):
        """ Resets the dialogue for an instance """
        bot_session.reset_dialogue()

    def respond(self, bot_session: 'BotSession', user_input: 'UserInput', **kwargs):
        """ Responds to a user input for a bot instance. """
        logger.info('#' * 100)
        logger.info(logcolor('user-utterance', '[INPUT] {}'.format(user_input)))
        logger.info('#' * 100)

        # give the bot the possibility to track all actions of this turn by adding an AccumulateDispatcher for this turn
        orig_dispatcher = bot_session.dispatcher
        acc_dispatcher = AccumulateDispatcher()
        bot_session.dispatcher = MultiDispatcher([orig_dispatcher, acc_dispatcher])

        initial_ds = bot_session.dialogue_state.clone()

        # check input
        if user_input.ignore():
            logger.info('Ignored input.')
        elif not user_input.is_valid():
            self._respond_invalid_input(bot_session, user_input, **kwargs)
        else:
            self._respond_valid_input(bot_session, user_input, **kwargs)

        # remove nsa dispatcher
        bot_session.dispatcher = orig_dispatcher

        if self.log_conversation:
            logger.info('Logging conversation into DB')
            self.log(bot_session, initial_ds, acc_dispatcher, user_input)

        logger.info('#' * 100)

    def _respond_invalid_input(self, bot_session: 'BotSession', user_input: 'UserInput', **kwargs):
        """ Respond to an invalid input """
        logger.debug('No valid input, using "utter_invalid_input" response')
        actions.utter_invalid_input(self).execute(bot_session)

    def _respond_no_intent_found(self, bot_session: 'BotSession', user_input: 'UserInput', **kwargs):
        """ Respond if no intent could be determined """
        consume_dialogue_state = True

        # check if a custom fallback intent is provided and execute if available
        if bot_session.iu_result.fallback_intent_id is not None:
            logger.debug('No confident intent found, using fallback intent "{}"'.format(
                bot_session.iu_result.fallback_intent_id))
            fallback_intent = self.intent(bot_session.iu_result.fallback_intent_id)

            # note that we do not re-extract & set entity-slots for a fallback intent

            # execute intent
            fallback_intent.action.execute(bot_session)

        elif isinstance(user_input, NLInput):
            logger.debug('No confident intent found, using "utter_intent_not_understand" response')
            actions.utter_intent_not_understand(self).execute(bot_session)

        else:
            logger.debug('No confident intent found, returning no response')
            consume_dialogue_state = False

        return consume_dialogue_state

    def _respond_valid_input(self, bot_session: 'BotSession', user_input: 'UserInput',
                             consume_dialogue_state: bool = True,
                             **kwargs
                             ):
        """ Respond to a valid input """

        # run IU
        logger.debug('Running IU...')
        iu_result = self.iu.run(user_input, bot_session.dialogue_state)
        iu_result.plog()

        # set IU result
        logger.debug('Set IU Result.')
        bot_session.iu_result = iu_result

        # entity-slots are set if intent is confident or verified

        # check if confident intent exist
        confidence_threshold = iu_result.confidence_threshold
        confident_intent_found = any([s.score >= confidence_threshold for s in iu_result.intent_ranking])

        # no intent found
        if not confident_intent_found:
            consume_dialogue_state = self._respond_no_intent_found(bot_session, user_input, **kwargs)
        else:
            # determine best candidate group of intents within confident intents
            confident_intents = [s for s in iu_result.intent_ranking if s.score >= confidence_threshold]

            top_confident_intents = top_similar_scores(confident_intents, key=lambda s: s.score,
                                                       min_score=confidence_threshold)
            logger.info(logcolor('intent', 'Top-Confident intents: {}'.format(top_confident_intents)))

            # check if there are intents to verify
            verify_intents = [s for s in top_confident_intents if self.intent(s.ref_id).verify]
            verify_intents = verify_intents[:self.max_verify_nr]
            logger.info(logcolor('intent', 'Verifiable intents:    {}'.format(verify_intents)))

            # not verify
            if len(verify_intents) <= 1:

                # set entity-slots
                logger.debug('Set Entity Slots.')
                bot_session.update_entity_slots(iu_result.entities)

                # execute action of best confident intent
                top_intent_id = iu_result.intent_ranking[0].ref_id
                intent = self.intent(top_intent_id)
                intent.action.execute(bot_session)

            else:
                # verify
                logger.info('Let user verify intents')
                verify_intent(self, verify_intents, iu_result, bot_session.dialogue_state, user_input).execute(
                    bot_session)

        # consume dialogue_state
        if consume_dialogue_state:
            logger.debug('Consume dialogue state')
            bot_session.dialogue_state.live()


    # OTHER ############################################################################################################

    def log(self, bot_session: 'BotSession', initial_ds: 'DialogueState', dispatcher: 'AccumulateDispatcher', user_input: 'UserInput'):

        self.db_handler.index_object(self.LOGGING_COLLECTION_NAME, {
            'time': datetime.datetime.now().isoformat(),
            'pid': os.getpid(),
            'tid': threading.get_ident(),
            'user_input': user_input.to_repr_dict(),
            'bot_responses': dispatcher.responses,
            'iu_result': bot_session.iu_result.to_repr_dict(),
            'initial_ds': initial_ds.to_repr_dict(),
            'final_ds': bot_session.dialogue_state.to_repr_dict()
        }, None)

        self.db_handler.commit(self.LOGGING_COLLECTION_NAME)
