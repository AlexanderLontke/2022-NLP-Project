from abc import ABC, abstractmethod
from dialogue_bot.models.inputs.nl import UserInput, NLInput
from code_search import CodeSearch, CodeSearchResult, RobertaCodeSearch
from paraphraser import Paraphraser, T5Paraphraser


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
            return code_search_result.code_string + "\n" + paraphrased_docstring
        else:
            return "I don't understand this kind of input."
