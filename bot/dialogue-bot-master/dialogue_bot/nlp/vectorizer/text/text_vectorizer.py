from abc import abstractmethod
from typing import List

import numpy as np
from dialogue_bot.nlp.vectorizer.word.word_vectorizer import vector_similarity


class TextVectorizer(object):
    def __init__(self, id: str):
        self.id = id

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError("Implement in Subclass")

    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        raise NotImplementedError("Implement in Subclass")

    @abstractmethod
    def vectorize_bulk(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError("Implement in Subclass")

    def vector_similarity(self, v1, v2, metric="cosine"):
        return vector_similarity(v1, v2, metric=metric)

    def similarity(self, text1: str, text2: str, metric="cosine"):
        v1, v2 = self.vectorize_bulk([text1, text2])
        return self.vector_similarity(v1, v2, metric=metric)

    @property
    def default_vector(self) -> np.ndarray:
        return np.zeros(self.dim)
