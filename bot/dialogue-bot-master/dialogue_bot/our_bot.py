from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.action import NLAction, NLPAction
from dialogue_bot.models.input import UserInput
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.triggers.nl import AnyNLTrigger
from dialogue_bot.models.response_generator import ResponseGenerator

bot = BotEnv("our_bot", "en")


class NlpResponseGenerator(ResponseGenerator):
    def generate_response(self, user_input: UserInput) -> str:
        if isinstance(user_input, NLInput):
            input_text: str = user_input.text
            return input_text + " " + input_text
        else:
            return "I don't understand this kind of input."


response_action = NLPAction("mirror", NlpResponseGenerator())

response_intent = Intent(
    bot,
    "test_intent",
    action=response_action,
    nl_trigger=AnyNLTrigger()
)

bot.register_intent(response_intent)

if __name__ == '__main__':
    bot.start(True)

    session = BotSession(bot)
    bot.respond(session, NLInput('I live in Germany'))
