from typing import Iterable


def default_enumeration(values: Iterable, str_func, cc_word: str):
    vs = list(values)
    if len(vs) <= 0:
        return ''
    elif len(vs) == 1:
        # "V1"
        return str_func(vs[0])
    elif len(vs) == 2:
        # "V1 and V2"
        return str_func(vs[0]) + ' ' + cc_word + ' ' + str_func(vs[-1])
    else:
        # "V1, V2 and V3"
        return ', '.join([str_func(v) for v in vs[:-1]]) + ' ' + cc_word + ' ' + str_func(vs[-1])


LANG_ENUMERATIONS = {
    'en': lambda values, str_func: default_enumeration(values, str_func, 'and'),
    'de': lambda values, str_func: default_enumeration(values, str_func, 'und'),
}