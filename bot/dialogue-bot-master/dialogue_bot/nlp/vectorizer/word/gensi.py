import logging
import os
from typing import List

import gensim
import numpy as np
from dialogue_bot import project_settings
from dialogue_bot.nlp.vectorizer.word.word_vectorizer import WordVectorizer

logger = logging.getLogger(__name__)


class GensimVectorizer(WordVectorizer):
    def __init__(
        self,
        id: str,
        model_path: str = os.path.join(
            project_settings.STATIC_DATA_DIRPATH, "GoogleNews-vectors-negative300.bin"
        ),
    ):
        super().__init__(id)
        self._model_path = model_path

        logger.info("Loading model from {}...".format(self._model_path))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            self._model_path, binary=True
        )

    @property
    def dim(self) -> int:
        return self.model.vector_size

    def vectorize(self, word: str) -> np.ndarray:
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            logger.warning('"{}" not in dictionary'.format(word))
            return self.default_vector

    def vectorize_bulk(self, words: List[str]) -> List[np.ndarray]:
        return [self.vectorize(word) for word in words]


if __name__ == "__main__":
    vectorizer = GensimVectorizer("GensimVectorizer")
    print(vectorizer.vectorize("car"))
