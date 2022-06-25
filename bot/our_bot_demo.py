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


function_explainer = FunctionExplainer()

# CODE SEARCH BOT
code_search_bot = BotEnv("code_search_bot", "en")

code_search_response_action = ResponseGeneratorAction(
    "code-search-response",
    CodeSearchResponseGenerator(function_explainer)
)
code_search_response_intent = Intent(
    code_search_bot, "code-search-intent",
    action=code_search_response_action,
    nl_trigger=AnyNLTrigger()
)
code_search_bot.register_intent(code_search_response_intent)

code_search_bot.start(True)
code_search_session = BotSession(code_search_bot)

# FUNCTION EXPLAINER BOT
function_explainer_bot = BotEnv("function_explainer_bot", "en")
# function_explainer_bot.register_value(
#     "FUNCTION_PYTHON",
#     regex_patterns=[r"(?i:[A-z]+\.[A-z]+\(.*\))"]
# )
# function_explainer_bot.register_entity(
#     "function",
#     value_refs=["FUNCTION_PYTHON"],
#     questions=["What does seaborn.pairplot() do?"]
# )

function_explainer_response_action = ResponseGeneratorAction(
    "function-explainer-response",
    FunctionExplainerResponseGenerator(function_explainer)
)

function_explainer_response_intent_name = "function-explainer-intent"
function_explainer_response_intent = Intent(
    function_explainer_bot, function_explainer_response_intent_name,
    action=function_explainer_response_action,
    nl_trigger=AnyNLTrigger()
)

function_explainer_bot.register_intent(
    function_explainer_response_intent
)
# function_explainer_bot.register_intent_phrase_patterns(
#     function_explainer_response_intent_name,
#     phrase_pattern=[
#         "Explain to me ((function))",
#         "Give me an explanation of the function ((function))",
#         "Help me understand ((function))",
#         "I want to better understand ((function))"
#     ],
# )

function_explainer_bot.start(True)
function_explainer_session = BotSession(function_explainer_bot)

app = FastAPI()


@app.post("/code-search")
def code_search_chat(chat_input: ChatInput):
    code_search_bot.respond(
        code_search_session,
        NLInput(chat_input.user_input)
    )


@app.post("/function-explanation")
def function_explainer_chat(chat_input: ChatInput):
    function_explainer_bot.respond(
        function_explainer_session,
        NLInput(chat_input.user_input)
    )
