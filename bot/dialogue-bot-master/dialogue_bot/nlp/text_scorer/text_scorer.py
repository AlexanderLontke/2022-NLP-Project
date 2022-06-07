from abc import ABC, abstractmethod

import numpy as np


def serialize_annotation(annotation: dict) -> dict:
    res = {}
    for k, v in annotation.items():
        class_name = v.__class__.__name__

        # numpy-arrays
        if isinstance(v, np.ndarray):
            v = v.tolist()

        res[k] = [class_name, v]

    return res


def deserialize_annotation(dct: dict) -> dict:
    res = {}
    for k, (class_name, v) in dct.items():

        # numpy-arrays
        if class_name == 'ndarray':
            v = np.array(v)

        res[k] = v
    return res


class TextScorer(ABC):
    def __init__(self, id: str):
        self.id = id

    @abstractmethod
    def annotate(self, text: str) -> dict:
        raise NotImplementedError('Implement in Subclass')

    @abstractmethod
    def similarity(self, annotation1: dict, annotation2: dict) -> float:
        raise NotImplementedError('Implement in Subclass')
