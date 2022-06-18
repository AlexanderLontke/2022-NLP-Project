import inspect
from abc import ABC, abstractmethod
from importlib import import_module

from dialogue_bot.models.inputs.nl import UserInput, NLInput
from code_search import CodeSearch, RobertaCodeSearch
from function_explainer import FunctionExplainer


class ResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, user_input: UserInput) -> str:
        pass


# IMPLEMENTATIONS

class CodeSearchResponseGenerator(ResponseGenerator):
    def __init__(self, function_explainer: FunctionExplainer):
        self.code_search: CodeSearch = RobertaCodeSearch()
        self.function_explainer = function_explainer

    def generate_response(self, user_input: UserInput) -> str:
        if isinstance(user_input, NLInput):
            input_text: str = user_input.text
            code_search_result: str = self.code_search.find_code_for_query(input_text)
            return self.function_explainer.explain_function(
                source_code=code_search_result
            )
        else:
            return "I don't understand this kind of input."


class FunctionExplainerResponseGenerator(ResponseGenerator):
    def __init__(self, function_explainer: FunctionExplainer):
        self.function_explainer = function_explainer

    def generate_response(self, user_input: UserInput) -> str:
        if isinstance(user_input, NLInput):
            input_text: str = user_input.text
            input_text = input_text[13:]
            module_name, function_name = input_text.split(".")
            module = import_module(module_name)
            method = getattr(module, function_name)
            source: str = inspect.getsource(method)
            return self.function_explainer.explain_function(
                source_code=source
            )
        else:
            return "I don't understand this kind of input."
