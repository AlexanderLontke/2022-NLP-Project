import logging
from typing import List

import numpy as np
from dialogue_bot.nlp.vectorizer.text.text_vectorizer import TextVectorizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerVectorizer(TextVectorizer):

    def __init__(self, id: str, model: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__(id)
        logger.info('Loading model {}...'.format(model))
        self.model = SentenceTransformer(model)

    @property
    def dim(self) -> int:
        return 768

    def vectorize(self, text: str) -> np.ndarray:
        return self.vectorize_bulk([text])[0]

    def vectorize_bulk(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(texts)
        return [embeddings[i] for i in range(embeddings.shape[0])]
