import logging
import typing
from abc import ABC
from typing import Set, List

from dialogue_bot.intent_understanding.iu import ExtractedEntity
from dialogue_bot.models.action import Action, NLAction, FuncAction
from dialogue_bot.models.input import UserInput
from dialogue_bot.models.state import DialogueState
from dialogue_bot.models.triggers.keyed import KeyedTrigger
from dialogue_bot.models.triggers.nl import NLTrigger, PhraseNLTrigger
from dialogue_bot.models.triggers.selection import SelectionTrigger
from dialogue_bot.models.utils import KeyComparable

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.bot_session import BotSession

logger = logging.getLogger(__name__)


class CustomScore(ABC):
    pass


class PerfectScore(CustomScore):
    """A perfect score is ranked on top of all other scores (e.g. any FloatScore)"""

    pass


class FloatScore(CustomScore):
    """A score given by a float value"""

    def __init__(self, value: float):
        self.value = value


class Intent(KeyComparable):
    """
    An intent captures an intention of the user.
    An intent stores many information about:
        - when it can be triggered (this is done using input_contexts)
        - how it can be triggered (e.g. through natural language, selections, keyed-inputs),
        - what happens once the intent is identified (see `Action`)
    These information are used by the intent-understanding (`IU`) unit which decides which intent is
    identified when a user input is received and which entities are extracted.
    While the intent-detection and entity extraction is primarily done by `IU`, each intent has also the possibility to
    override the results of `IU`. This can be done by overrinding the intent-methods `custom_score` or `custom_extract`.
    This can be used for special intents that may know themselves how they should be identified or extract entities.
    """

    DEFAULT_DOMAIN = "default"

    def __init__(
        self,
        env: "BotEnv",
        id: str,
        input_contexts: List[str] = None,
        domains: List[str] = None,
        nl_trigger: "NLTrigger" = None,
        selection_trigger: "SelectionTrigger" = None,
        keyed_trigger: "KeyedTrigger" = None,
        action: "Action" = None,
        verify: bool = True,
        verify_description: str = None,
        entity_filter: List[str] = None,
    ):
        """
        Creates an intent.
        :param env:                   The Bot Environment this intent belongs to
        :param id:                    A unique id for the intent
        :param domains:               The domains this intent is attached to. Can be used to group intents together.
        :param input_contexts:        The required input contexts for this intent
        :param nl_trigger:            [Optional] A NLTrigger.
        :param selection_trigger:     [Optional] A SelectionTrigger
        :param keyed_trigger:         [Optional] A KeyedTrigger
        :param action:                [Optional] The action that is executed once this intent is selected
        :param verify:                Specifies if this intent can be verified by the user
        :param verify_description:    The verification description that is used in order to verify this intent
            e.g. "Say hello" (for "hello" intent) or "Ask chatbot where to find shops" (for "find_shops" intent)
        :param entity_filter:         [Optional] The entities that can be extracted for that intent. If None, then all
            entities are considered.
        """
        if domains is None:
            domains = {self.DEFAULT_DOMAIN}
        if input_contexts is None:
            input_contexts = []

        self.env = env
        self.id: str = id
        self.domains = set(domains)
        self.input_contexts = set(input_contexts)
        self.nl_trigger = nl_trigger
        self.selection_trigger = selection_trigger
        self.keyed_trigger = keyed_trigger
        self._action = action
        self.verify = verify
        self._verify_description = verify_description
        self.entity_filter = set(entity_filter) if entity_filter is not None else None

    def has_contexts(self, dialogue_state: "DialogueState") -> bool:
        """Determines if the required input contexts are present in the dialogue state"""
        if not all([c in dialogue_state.context_names for c in self.input_contexts]):
            return False
        return True

    def has_trigger(self, user_input: "UserInput") -> bool:
        """Determines if the intent has the required triggers to process the user_input"""
        triggers = [self.nl_trigger, self.selection_trigger, self.keyed_trigger]
        return any([t.matches_input(user_input) for t in triggers if t is not None])

    def custom_score(
        self,
        user_input: "UserInput",
        dialogue_state: "DialogueState",
        score: "CustomScore",
    ) -> "CustomScore":
        """Use a custom score for the intent"""
        return score

    def custom_extract(
        self,
        user_input: "UserInput",
        dialogue_state: "DialogueState",
        entities: Set["ExtractedEntity"],
    ) -> Set["ExtractedEntity"]:
        """Allow the intent to (re-)extract entities"""
        return entities

    def _do_execute(self, session: "BotSession"):
        """Can be implemented instead of passing an action function to the constructor"""
        NLAction("No response defined").execute(session)

    @property
    def action(self) -> "Action":
        """The action that should be executed when the intent is chosen"""
        if self._action is None:
            return FuncAction(self._do_execute, name="{}-Action".format(self.id))

        return self._action

    @action.setter
    def action(self, action):
        self._action = action

    @property
    def verify_description(self) -> str:
        if self._verify_description is not None:
            return self._verify_description
        elif (
            (self.nl_trigger is not None)
            and isinstance(self.nl_trigger, PhraseNLTrigger)
            and (len(self.nl_trigger.expression.phrase_patterns) > 0)
        ):
            return (
                '"' + list(self.nl_trigger.expression.phrase_patterns)[0].pattern + '"'
            )
        else:
            return "[Intent {}]".format(self.id)

    def key_tuple(self) -> tuple:
        return (self.id,)

    def __repr__(self):
        return '({}: "{}", domains={}, input_contexts={}, nl_trigger={}, selection_trigger={}, keyed_trigger={})'.format(
            self.__class__.__name__,
            self.id,
            self.domains,
            self.input_contexts,
            self.nl_trigger,
            self.selection_trigger,
            self.keyed_trigger,
        )
