from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.triggers.nl import AnyNLTrigger

from fastapi import FastAPI
from pydantic import BaseModel
from response_generator import CodeSearchResponseGenerator
from response_generator_action import ResponseGeneratorAction


class ChatInput(BaseModel):
    user_input: str


bot = BotEnv("our_bot", "en")
code_search_response_action = ResponseGeneratorAction("code-search-response", CodeSearchResponseGenerator())
response_intent = Intent(
    bot, "code-search-intent", action=code_search_response_action, nl_trigger=AnyNLTrigger()
)
bot.register_intent(response_intent)

bot.start(True)
session = BotSession(bot)

app = FastAPI()


@app.post("/chat")
def read_root(chat_input: ChatInput):
    bot.respond(session, NLInput(chat_input.user_input))
