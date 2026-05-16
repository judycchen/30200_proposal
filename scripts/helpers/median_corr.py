# Code by Ryleigh Nash


import numpy as np


def median_corr(corr_array, axis=0):

    """ 
    compute median correlation.
    input: 
    - array of corr scores
    - axis to take median across
    output: median corr scores
    """

    return np.median((corr_array), axis)

