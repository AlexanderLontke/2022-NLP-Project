from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.triggers.nl import AnyNLTrigger

from fastapi import FastAPI
from pydantic import BaseModel
from response_generator import CodeSearchResponseGenerator, FunctionExplainerResponseGenerator
from response_generator_action import ResponseGeneratorAction
from function_explainer import FunctionExplainer


class ChatInput(BaseModel):
    user_input: str


bot = BotEnv("our_bot", "en")
function_explainer = FunctionExplainer()
code_search_response_action = ResponseGeneratorAction(
    "code-search-response",
    CodeSearchResponseGenerator(function_explainer)
)
code_search_response_intent = Intent(
    bot, "code-search-intent", action=code_search_response_action, nl_trigger=AnyNLTrigger()
)
bot.register_intent(code_search_response_intent)

function_explainer_response_action = ResponseGeneratorAction(
    "function-explainer-response",
    FunctionExplainerResponseGenerator(function_explainer)
)

function_explainer_response_intent_name = "function-explainer-intent"
function_explainer_response_intent = Intent(
    bot, function_explainer_response_intent_name,
    action=code_search_response_action
)

bot.register_intent(function_explainer_response_intent)
bot.register_intent_phrase_patterns(
    function_explainer_response_intent_name,
    phrase_pattern="Explain to me function x",
)

bot.start(True)
session = BotSession(bot)

app = FastAPI()


@app.post("/chat")
def read_root(chat_input: ChatInput):
    bot.respond(session, NLInput(chat_input.user_input))
