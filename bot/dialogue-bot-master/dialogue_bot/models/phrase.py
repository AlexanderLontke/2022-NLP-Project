from typing import List

from dialogue_bot.models.utils import KeyComparable


class PhraseEntity(object):
    def __init__(self, start: int, end: int, entity: str, value: str):
        """
        An entity-annotation.
        :param start:       The start-position of the annotated span in the containing text
        :param end:         The end-position of the annotated span in the containing text
        :param entity:      The entity-class
        :param value:       The entity-value-class
        """
        if end <= start:
            raise ValueError('end cannot be <= start')
        self.start = start
        self.end = end
        self.entity = entity
        self.value = value

    def __repr__(self):
        return '({}={})'.format(self.entity, self.value)


class Phrase(KeyComparable):
    """
    A training phrase.

    {
      "class_id": "book_flight",
      "text": "Book a flight from Berlin to SF",
      "entities": [
        {
          "start": 19,
          "end": 25,
          "entity": "city",
          "value": "BE",
          "extractor": "unknown",
        },
        {
          "start": 29,
          "end": 31,
          "entity": "city",
          "value": "SF",
          "extractor": "unknown",
        }
      ]
    }
    """

    def __init__(self, expression_id: str, text: str,
                 entities: List[PhraseEntity] = None):
        if entities is None:
            entities = []

        self.expression_id = expression_id
        self.text = text
        self.entities = entities
        for entity in entities:
            self.annotate(entity)

    def key_tuple(self) -> tuple:
        return (
            self.__class__.__name__,
            self.expression_id,
            self.text
        )

    def annotate(self, entity: 'PhraseEntity'):
        # assert that self.entities are in order
        before = [e for e in self.entities if e.end <= entity.start]
        after = [e for e in self.entities if e.start >= entity.end]

        # can insert?
        if len(before) + len(after) == len(self.entities):
            self.entities = before + [entity] + after

    def __repr__(self):
        entity_repr = lambda e: '({}={}:"{}")'.format(e.entity, e.value, self.text[e.start:e.end])
        return '("{}" | {})'.format(self.text, ' | '.join([entity_repr(e) for e in self.entities]))


if __name__ == '__main__':
    phrase = Phrase(None, 'I live in Switzerland and Germany')
    phrase.annotate(PhraseEntity(10, 21, 'country', 'CH'))
    phrase.annotate(PhraseEntity(26, 35, 'country', 'DE'))
    print(phrase)
