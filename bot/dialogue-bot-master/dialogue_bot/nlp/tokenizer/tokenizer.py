from abc import ABC
from typing import List, Tuple


class WordTokenizer(ABC):
    def __init__(self, id: str):
        self.id = id

    def tokenize_spans(self, text: str) -> List[Tuple[int, int]]:
        pass

    def tokenize(self, text: str) -> List[str]:
        return [text[fr:to] for fr, to in self.tokenize_spans(text)]
