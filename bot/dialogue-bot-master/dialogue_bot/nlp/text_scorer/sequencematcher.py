import difflib

from dialogue_bot.nlp.text_scorer.text_scorer import TextScorer


class SequencematcherTextScorer(TextScorer):

    def attributes(self) -> dict:
        return {
            'class': self.__class__.__name__,
        }

    def annotate(self, text: str) -> dict:
        return {'text': text}

    def similarity(self, annotation1: dict, annotation2: dict) -> float:
        seq = difflib.SequenceMatcher(None, annotation1['text'].lower(), annotation2['text'].lower())
        return seq.ratio()
