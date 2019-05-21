import pickle as pkl

import numpy as np
import pybobyqa


def wrapper_pybobyqa(crit_func, start_values, **kwargs):

    try:
        pybobyqa.solve(crit_func, start_values, **kwargs)
    except StopIteration:
        pass

    return pkl.load(open("monitoring.estimagic.pkl", "rb"))
