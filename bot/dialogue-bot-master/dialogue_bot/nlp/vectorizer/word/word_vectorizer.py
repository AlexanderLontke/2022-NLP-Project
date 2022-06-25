from abc import ABC, abstractmethod
from typing import List

import numpy as np


def vector_similarity(
    vector1: np.ndarray, vector2: np.ndarray, metric="cosine"
) -> float:
    if metric == "cosine":
        # cosine distance
        if np.linalg.norm(vector1) * np.linalg.norm(vector2) == 0:
            return 0
        return (
            np.dot(vector1, vector2)
            * 1.0
            / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        )
    else:
        raise ValueError('Metric "{}" not supported'.format(metric))


class WordVectorizer(ABC):
    def __init__(self, id: str):
        self.id = id

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError("Implement in Subclass")

    @abstractmethod
    def vectorize(self, word: str) -> np.ndarray:
        raise NotImplementedError("Implement in Subclass")

    @abstractmethod
    def vectorize_bulk(self, words: List[str]) -> List[np.ndarray]:
        raise NotImplementedError("Implement in Subclass")

    def vector_similarity(self, v1, v2, metric="cosine"):
        return vector_similarity(v1, v2, metric=metric)

    def similarity(self, word1: str, word2: str, metric="cosine"):
        v1, v2 = self.vectorize_bulk([word1, word2])
        return self.vector_similarity(v1, v2, metric=metric)

    @property
    def default_vector(self) -> np.ndarray:
        return np.zeros(self.dim)
