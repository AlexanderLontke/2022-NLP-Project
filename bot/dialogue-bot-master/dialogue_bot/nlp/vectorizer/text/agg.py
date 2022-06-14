import typing
from typing import List, Iterable

import numpy as np
from dialogue_bot.nlp.vectorizer.text.text_vectorizer import TextVectorizer

if typing.TYPE_CHECKING:
    from dialogue_bot.nlp.vectorizer.word.word_vectorizer import WordVectorizer
    from dialogue_bot.nlp.tokenizer.tokenizer import WordTokenizer


class AggTextVectorizer(TextVectorizer):
    def __init__(
        self,
        id: str,
        word_tokenizer: "WordTokenizer",
        word_vectorizer: "WordVectorizer",
        remove_stopwords: bool = False,
        stopwords: Iterable = None,
    ):
        super().__init__(id)
        self.word_tokenizer = word_tokenizer
        self.word_vectorizer = word_vectorizer
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords

    @property
    def dim(self) -> int:
        return self.word_vectorizer.dim

    def vectorize(self, text: str) -> np.ndarray:
        return self.vectorize_bulk([text])[0]

    def _rm_stopwords(self, words):
        # for the first token ignore cases (e.g. "And for Germany?")
        return [
            w
            for i, w in enumerate(words)
            if (i == 0 and w.lower() not in [s.lower() for s in self.stopwords])
            or (i > 0 and w not in self.stopwords)
        ]

    def vectorize_bulk(self, texts: List[str]) -> List[np.ndarray]:
        res = []
        for text in texts:
            if self.remove_stopwords:
                words = self._rm_stopwords(self.word_tokenizer.tokenize(text))
            else:
                words = [w for w in self.word_tokenizer.tokenize(text)]

            if len(words) > 0:
                vectors = self.word_vectorizer.vectorize_bulk(words)
                matrix = np.vstack(vectors)
                res.append(matrix.mean(0))
            else:
                res.append(self.default_vector)

        return res

    @property
    def default_vector(self) -> np.ndarray:
        return np.zeros(self.dim)
