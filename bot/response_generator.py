import ast

from importlib import import_module
from typing import List, Optional
from abc import ABC, abstractmethod

import inspect
from dialogue_bot.models.inputs.nl import UserInput, NLInput
from code_search import CodeSearch, CodeSearchResult, RobertaCodeSearch
from paraphraser import Paraphraser, T5Paraphraser


class FirstFunctionResult:
    def __init__(self, function_name, function_docstring):
        self.function_name = function_name
        self.function_docstring = function_docstring


def _find_first_module_name(module_name) -> str:
    if isinstance(module_name, ast.Name):
        result = getattr(module_name, "id")
        return result
    elif isinstance(module_name, ast.Attribute):
        return _find_first_module_name(module_name.value)
    elif isinstance(module_name, ast.Call):
        return _find_first_module_name(module_name.func)


def _parse_first_function_from_method(source_code: str) -> Optional[FirstFunctionResult]:
    abstract_syntax_tree = ast.parse(source=source_code)
    call_objects: List[ast.Call] = [node for node in ast.walk(abstract_syntax_tree) if isinstance(node, ast.Call)]
    for call in call_objects:
        attributes_of_first_call = []

        for node in ast.walk(call):
            if isinstance(node, ast.Attribute):
                attributes_of_first_call.append(getattr(node, "attr"))
                break  # At the moment we only support methods of the format module.function
        function_call_segments = list(reversed(attributes_of_first_call))
        module_name = _find_first_module_name(call)

        print("Module:", module_name)
        print("Function call segments:", print(function_call_segments))

        if module_name == "np":
            module_name = "numpy"
        elif module_name == "pd":
            module_name = "pandas"
        elif module_name == "sns":
            module_name = "seaborn"
        elif module_name == "plt":
            module_name = "matplotlib.pyplot"
        try:
            function_name = function_call_segments[0]
            module = import_module(module_name)
            method = getattr(module, function_name)
            source = inspect.getsource(method)
            docstring = ast.get_docstring(ast.parse(source).body[0])
            return FirstFunctionResult(module_name + "." + function_name, docstring)
        except IOError:
            continue
        except AttributeError:
            continue
        except ModuleNotFoundError:
            continue
        except TypeError:
            continue
        except IndexError:
            continue
    return None


class ResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, user_input: UserInput) -> str:
        pass


# IMPLEMENTATIONS

class CodeSearchResponseGenerator(ResponseGenerator):
    def __init__(self):
        self.code_search: CodeSearch = RobertaCodeSearch()
        self.paraphraser: Paraphraser = T5Paraphraser()

    def generate_response(self, user_input: UserInput) -> str:
        if isinstance(user_input, NLInput):
            input_text: str = user_input.text
            code_search_result: CodeSearchResult = self.code_search.find_code_for_query(input_text)
            paraphrased_docstring = self.paraphraser.paraphrase(code_search_result.doc_string)

            first_function_result: FirstFunctionResult = _parse_first_function_from_method(
                code_search_result.code_string)
            if first_function_result:
                paraphrased_deepdive_docstring = self.paraphraser.paraphrase(first_function_result.function_docstring)
                deepdive_explanation = f"The first method for which I could create an explanation is " \
                                       f"{first_function_result.function_name}." \
                                       f"An explanation of this method is {paraphrased_deepdive_docstring}"
            else:
                deepdive_explanation = "I was unable to provide a deeper explanation."
            return code_search_result.code_string + "\n" + paraphrased_docstring + "\n\n" + deepdive_explanation
        else:
            return "I don't understand this kind of input."
