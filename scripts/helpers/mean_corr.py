# Code by Shan Gao


import numpy as np


def mean_corr(corr_array, axis=0):

    """ 
    compute mean correlation with fisher's z transform.
    input: 
    - array of corr scores to be averaged
    - axis to take average across
    output: averaged corr scores
    """

    return np.tanh(np.nanmean(np.arctanh(corr_array), axis))

