import hashlib

import jsonpickle


def strict_crypto_hash(obj, limit: int = None) -> str:
    """ Order of dictionary-keys is important """
    s = jsonpickle.encode(obj)
    s = hashlib.sha1(s.encode("UTF-8")).hexdigest()
    if limit is not None:
        s = s[:limit]
    return s