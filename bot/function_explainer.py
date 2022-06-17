import ast
from paraphraser import Paraphraser, T5Paraphraser

from dill.source import getsource


def _parse_docstring_from_sourcecode(source_code: str) -> str:
    # Parse the docstring of a python function from its source code
    docstring = ast.get_docstring(ast.parse(source_code).body[0])
    return docstring.replace('\n', "")


class FunctionExplainer:
    def __init__(self):
        self.paraphraser: Paraphraser = T5Paraphraser()

    def explain_function(self, function_name: str, source_code: str = None) -> str:
        if source_code is None:
            try:
                source_code = getsource(globals()[function_name])
            except KeyError:
                return "I could not explain this function since I dont have access to it's source code."

        doc_string = _parse_docstring_from_sourcecode(source_code)
        return self._paraphrase_doc_string(doc_string)

    def _paraphrase_doc_string(self, doc_string: str) -> str:
        return self.paraphraser.paraphrase(doc_string)


if __name__ == '__main__':
    print(getsource(globals()["T5Paraphraser"]))
