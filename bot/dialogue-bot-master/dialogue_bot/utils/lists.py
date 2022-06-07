import statistics

import numpy as np


def list_aggregate_func(function):
    if function == 'max':
        return lambda lst: max(lst)
    if function == 'min':
        return lambda lst: min(lst)
    if function == 'mean':
        return lambda lst: np.mean(lst)
    if function == 'sum':
        return lambda lst: sum(lst)
    if function == 'median':
        return lambda lst: statistics.median(lst)
    else:
        raise ValueError('Invalid function')