from dialogue_bot.models.action import Action
from dialogue_bot.bot_session import BotSession
from response_generator import ResponseGenerator


class ResponseGeneratorAction(Action):
    def __init__(self, name, response_generator: ResponseGenerator):
        self._name = name
        self._response_generator = response_generator

    @property
    def name(self):
        return self._name

    def _do_execute(self, session: "BotSession"):
        user_input = session.iu_result.user_input
        response = self._response_generator.generate_response(user_input=user_input)
        session.dispatcher.utter(session, response)
