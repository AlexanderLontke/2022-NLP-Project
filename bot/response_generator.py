import re
import inspect

from abc import ABC, abstractmethod
from importlib import import_module

from dialogue_bot.models.inputs.nl import UserInput, NLInput
from code_search import CodeSearch, RobertaCodeSearch
from function_explainer import FunctionExplainer, get_module_and_function


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
            python_function_pattern = r"([A-z]+)\.([A-z]+)\(.*\)"
            match_object = re.search(python_function_pattern, input_text)
            if match_object is None:
                return "Please specify method using the following pattern: module.function()"
            module_name = match_object.group(1)
            function_name = match_object.group(2)
            source: str = get_module_and_function(function_name, module_name)
            if source is None:
                return "I could not find the specified module and function."
            else:
                return self.function_explainer.explain_function(
                    source_code=source
                )
        else:
            return "I don't understand this kind of input."
