import typing

from dialogue_bot.nlp.text_scorer.text_scorer import TextScorer

if typing.TYPE_CHECKING:
    from dialogue_bot.nlp.vectorizer.text.text_vectorizer import TextVectorizer


class VectorTextScorer(TextScorer):
    def __init__(self, id: str, vectorizer: "TextVectorizer", ignore_case: bool):
        super().__init__(id)
        self.vectorizer = vectorizer
        self.ignore_case = ignore_case

    def annotate(self, text: str) -> dict:
        preprocess_text = lambda s: s.lower() if self.ignore_case else s

        return {"vector": self.vectorizer.vectorize(preprocess_text(text))}

    def similarity(self, annotation1: dict, annotation2: dict) -> float:
        return self.vectorizer.vector_similarity(
            annotation1["vector"], annotation2["vector"]
        )
