from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.intent import Intent
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.triggers.nl import AnyNLTrigger, FallbackNLTrigger
from dialogue_bot.models.entity import Entity

from fastapi import FastAPI
from pydantic import BaseModel
from response_generator import CodeSearchResponseGenerator, FunctionExplainerResponseGenerator
from response_generator_action import ResponseGeneratorAction
from function_explainer import FunctionExplainer


class ChatInput(BaseModel):
    user_input: str


bot = BotEnv("our_bot", "en")

bot.register_value("FUNCTION_PYTHON", regex_patterns=[r"(?i:[A-z]+\.[A-z]+\(.*\))"])
bot.register_entity(
    "function",
    value_refs=["FUNCTION_PYTHON"],
    questions=["What does seaborn.pairplot() do?"]
)
# function_explainer = FunctionExplainer()
# code_search_response_action = ResponseGeneratorAction(
#     "code-search-response",
#     CodeSearchResponseGenerator(function_explainer)
# )
code_search_response_intent = Intent(
    bot, "code-search-intent",
    # action=code_search_response_action,
    nl_trigger=FallbackNLTrigger()
)
bot.register_intent(code_search_response_intent)

# function_explainer_response_action = ResponseGeneratorAction(
#     "function-explainer-response",
#     FunctionExplainerResponseGenerator(function_explainer)
# )

function_explainer_response_intent_name = "function-explainer-intent"
function_explainer_response_intent = Intent(
    bot, function_explainer_response_intent_name,
    # action=code_search_response_action
)

bot.register_intent(function_explainer_response_intent)
bot.register_intent_phrase_patterns(
    function_explainer_response_intent_name,
    phrase_pattern=[
        "Explain to me ((function))",
        "Give me an explanation of the function ((function))",
        "Help me understand ((function))",
        "I want to better understand ((function))"
    ],
)

bot.start(True)
session = BotSession(bot)

app = FastAPI()


@app.post("/chat")
def chat(chat_input: ChatInput):
    bot.respond(session, NLInput(chat_input.user_input))
