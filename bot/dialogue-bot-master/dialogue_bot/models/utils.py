import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List


class KeyComparable(ABC):
    @abstractmethod
    def key_tuple(self) -> tuple:
        raise NotImplementedError("Implement in subclass")

    def __hash__(self) -> int:
        """Attention: This is no cryptographic hash (hashes might differ from session to session)"""
        return hash(self.key_tuple())

    def crypto_hash(self) -> str:
        s = json.dumps(self.key_tuple(), sort_keys=True)
        return hashlib.sha1(s.encode("UTF-8")).hexdigest()[:20]

    def __eq__(self, other):
        return self.key_tuple() == other.key_tuple()


class RankingScore(object):
    def __init__(self, ref_id: str, score: float):
        self.ref_id = ref_id
        self.score = score

    def __repr__(self):
        return '("{}": {:.3f})'.format(self.ref_id, self.score)

    def to_repr_dict(self) -> dict:
        return self.__dict__


def islist(obj):
    return isinstance(obj, list) or isinstance(obj, set)


def chunk_list(lst: list, max_elems: int):
    """Splits a (potential large) list into smaller sublists of the given maximum size"""
    res = []
    for i in list(range(len(lst)))[::max_elems]:
        res.append(lst[i : i + max_elems])
    return res


def complete_ranking(
    ranking: List["RankingScore"], ids: List, added_conf=0.0
) -> List["RankingScore"]:
    """Will add or remove RankingScores until only RankingScores for ``ids`` exist"""

    res = []
    res_ids = set([])
    for s in ranking:
        if s.ref_id not in ids:
            continue
        res.append(s)
        res_ids.add(s.ref_id)
    for id in ids:
        if id in res_ids:
            continue
        res.append(RankingScore(id, added_conf))
        res_ids.add(id)

    return res


def stratify_train_test_split(X, label_func, label_test_split=0.33):
    """
    Splits each label by the given `label_test_split` ratio.
    The test split will round up, meaning that if there are at least two label instances and label_test_split > 0,
    then there will be a label-instance in the test set.
    """

    # group by label
    grouped = defaultdict(set)
    for x in X:
        grouped[label_func(x)].add(x)

    train_X, test_X = [], []

    # split for each label
    for label, group in grouped.items():
        group = list(group)
        if len(group) == 0:
            pass
        elif len(group) == 1:
            train_X.append(group[0])
        else:
            test_amount = math.ceil(len(group) * label_test_split)

            # shuffle split
            random.shuffle(group)
            train_X.extend(group[test_amount:])
            test_X.extend(group[:test_amount])

    return train_X, test_X
