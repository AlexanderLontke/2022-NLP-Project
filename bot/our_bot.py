from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.triggers.nl import AnyNLTrigger

from response_generator import CodeSearchResponseGenerator
from response_generator_action import ResponseGeneratorAction

bot = BotEnv("our_bot", "en")


code_search_response_action = ResponseGeneratorAction("code-search-response", CodeSearchResponseGenerator())

response_intent = Intent(
    bot, "code-search-intent", action=code_search_response_action, nl_trigger=AnyNLTrigger()
)

bot.register_intent(response_intent)

if __name__ == "__main__":
    bot.start(True)

    session = BotSession(bot)
    bot.respond(session, NLInput("I live in Germany"))
