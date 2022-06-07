import logging
import random
import typing
from abc import abstractmethod, ABC

from dialogue_bot.models.context import TTLContext
from dialogue_bot.models.utils import islist
from dialogue_bot.models.utter_phrase import UtterPhrase

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_session import BotSession

logger = logging.getLogger(__name__)


class Action(ABC):
    """
    Defines an Interface for (predefined) Actions, which can be used to simplify the implementation of intents.
    The execution of actions is logged during runtime, which makes them very easy to trace.
    Therefore, the use of actions is highly recommended!
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _do_execute(self, session: 'BotSession'):
        pass

    def execute(self, session: 'BotSession'):
        logger.debug('Execute action "{}"'.format(self.name))
        session.dispatcher.before_action(session, self)
        self._do_execute(session)
        session.dispatcher.after_action(session, self)

    def __repr__(self):
        return '({}: "{}")'.format(self.__class__.__name__, self.name)


# IMPLEMENTATIONS ######################################################################################################


class NoneAction(Action):
    """An action that just enjoys the simplicity of life and does nothing"""

    def __init__(self):
        pass

    @property
    def name(self):
        return 'NoneAction'

    def _do_execute(self, session: 'BotSession'):
        pass


class FuncAction(Action):
    """Converting an action method into an action instance"""

    def __init__(self, action_func, name: str = None):
        if name is None:
            name = 'FuncAction'
        self._name = name
        self._action_func = action_func

    @property
    def name(self):
        return self._name

    def _do_execute(self, session: 'BotSession'):
        self._action_func(session)


class ExecuteIntentAction(Action):
    """Executes the specified intent"""
    def __init__(self, intent_id: str):
        self._intent_id = intent_id

    @property
    def name(self):
        return 'execute_intent("{}")'.format(self._intent_id)

    def _do_execute(self, session: 'BotSession'):
        session.env.intent(self._intent_id).action.execute(session)


class CustomAction(Action):
    """Executes a custom action specified by a name"""
    def __init__(self, name: str, **kwargs):
        self._name = name
        self._args = kwargs

    @property
    def name(self):
        return 'custom({})'.format(self._name)

    def _do_execute(self, session: 'BotSession'):
        session.dispatcher.custom(session, self._name, **self._args)


class NLAction(Action):
    """Returns a natural language response"""

    def __init__(self, text_choices, default_render_func=lambda entity: None, **kwargs):
        """If more than one response is passed, then those responses are seen as possible alternatives"""

        if not islist(text_choices):
            text_choices = [text_choices]

        text_choices = [(UtterPhrase(r) if not isinstance(r, UtterPhrase) else r) for r in text_choices]

        self._text_choices = text_choices.copy()
        self._default_render_func = default_render_func
        self._args = kwargs

    @property
    def name(self):
        return 'nl("{}")'.format(self._text_choices[0].text)

    def _do_execute(self, session: 'BotSession'):
        # render utterance

        slots = session.dialogue_state.slots

        # select a response (prioritize responses that have the slots needed to be rendered)
        text_choices = self._text_choices.copy()
        random.shuffle(text_choices)
        text_choices = sorted([c for c in text_choices], key=lambda c: c.render_missing(slots), reverse=False)

        response = text_choices[0].render(session.env, slots, default_render_func=self._default_render_func)
        session.dispatcher.utter(session, response, **self._args)


class Choice(object):
    """
    A choice of a selection.
    A choice is uniquely identified by a key and an index (idx).
    The index can be used when there are multiple choices with the same key.
    """
    def __init__(self, key: str, idx: int, text: str, **kwargs):
        self.key = key
        self.idx = idx
        self.text = text
        self.args = kwargs

    def __repr__(self):
        return '(Choice: {}:{}, "{}")'.format(self.key, self.idx, self.text)


class ChoiceAction(Action):
    """Returns a list of choices for the user"""

    def __init__(self, choices, text: str = None, default_render_func=lambda entity: None, **kwargs):
        if not islist(choices):
            choices = [choices]

        self._choices = choices.copy()
        self._text = text
        self._default_render_func = default_render_func
        self._args = kwargs

    @property
    def name(self):
        return 'choice({})'.format(self._choices)

    def _do_execute(self, session: 'BotSession'):
        # render choices

        slots = session.dialogue_state.slots
        choices = [
            Choice(c.key, c.idx,
                   UtterPhrase(c.text).render(session.env, slots, default_render_func=self._default_render_func),
                   **c.args) for c in self._choices]
        session.dispatcher.choice(session, choices, text=self._text, **self._args)


class MediaAction(Action):
    """ Displays a media object for the user specified by media_id"""
    def __init__(self, media_id: str, media_type: str = None, title: str = None, desc: str = None, **kwargs):
        self.media_id = media_id
        self.media_type = media_type
        self.title = title
        self.desc = desc
        self.args = kwargs

    @property
    def name(self):
        return 'media({})'.format(self.media_id)

    def _do_execute(self, session: 'BotSession'):
        session.dispatcher.media(session, self.media_id, media_type=self.media_type, title=self.title, desc=self.desc,
                                 **self.args)


class SlotSetAction(Action):
    """Sets a value for a slot"""

    def __init__(self, slot: str, value):
        self._slot = slot
        self._value = value

    @property
    def name(self):
        return 'slot_set({}: {})'.format(self._slot, self._value)

    def _do_execute(self, session: 'BotSession'):
        session.dialogue_state.set_slot(self._slot, self._value)


class SlotClearAction(Action):
    """Deletes a slot value"""

    def __init__(self, slot: str):
        self._slot = slot

    @property
    def name(self):
        return 'slot_clear({})'.format(self._slot)

    def _do_execute(self, session: 'BotSession'):
        session.dialogue_state.clear_slot(self._slot)


class ContextSetAction(Action):
    """Sets a context"""

    def __init__(self, name: str, **kwargs):
        self._context = TTLContext(name, **kwargs)

    @property
    def name(self):
        return 'context_set("{}", remaining={})'.format(self._context.name, self._context.remaining)

    def _do_execute(self, session: 'BotSession'):
        session.dialogue_state.set_context(self._context)


class ContextClearAction(Action):
    """Deletes a context"""

    def __init__(self, name: str):
        self._context_name = name

    @property
    def name(self):
        return 'context_clear("{}")'.format(self._context_name)

    def _do_execute(self, session: 'BotSession'):
        session.dialogue_state.clear_context(self._context_name)


# class ReRunTextUtteranceAction(Action):
#     def __init__(self, utterance: str, domain):
#         self._utterance = TextUtterance(utterance)
#         self._utterance.domain = domain
#
#     @property
#     def name(self):
#         return 'rerun({})'.format(self._utterance)
#
#     def do_execute(self, bot: 'BotEnv', session: 'BotSession'):
#         bot.respond(user_id, self._utterance, consume_dialogue_state=False)


class JoinedAction(Action):
    """An action that executes several actions in the given order"""

    def __init__(self, actions: list, name: str = None):
        if name is None:
            name = 'JoinedAction'
        self._name = name
        self._actions = actions

    @property
    def name(self):
        return self._name

    def _do_execute(self, session: 'BotSession'):
        for action in self._actions:
            action.execute(session)


class DefaultNLAction(JoinedAction):
    """ Returns a natural language response and sets contexts after that"""
    def __init__(self, responses, contexts=None, **kwargs):
        if contexts is None:
            contexts = []

        if not islist(responses):
            responses = [responses]
        if not islist(contexts):
            contexts = [contexts]

        contexts = [((x, None) if isinstance(x, str) else x) for x in contexts]

        actions = [NLAction(response) for response in responses] + \
                  [ContextSetAction(n, lifetime=l) for n, l in contexts]
        super().__init__(actions, **kwargs)



