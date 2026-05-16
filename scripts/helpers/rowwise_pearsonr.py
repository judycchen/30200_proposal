# Code by Shan Gao


import numpy as np
from scipy.stats import pearsonr


def rowwise_pearsonr_one_to_one(matrix_a, matrix_b):
   
    """" 
    inputs: 2d matrix_a and matrix_b of same shape (region * time)
    outputs: 1d array of pearson's r between row i in matrix_a and row i in matrix_b
    """

    corr_per_row = [pearsonr(matrix_a[i], matrix_b[i])[0] for i in range(len(matrix_a))]
    return np.array(corr_per_row)



def rowwise_pearsonr_many_to_one(many_instances, one_instance, metric=None):

    """ 
    inputs: time series data for **a single region**
    - many_instances: 2d array, obs * time
    - one_instance: 1d array of length time
    - metric: take median or mean across subjs to get the summary corr across obs
    outputs: 
    - if metric==None: 1d array of length num obs, storing each obs' corr with one_instance
    - if metric=='median'/'mean': summary corr across obs (scalar)
    """

    corr_allinstances = []
    for instance in many_instances:
        corr_allinstances.append(pearsonr(instance, one_instance)[0])

    if not metric:
        return np.array(corr_allinstances)
    elif metric == 'median':
        return np.median(corr_allinstances)
    elif metric == 'mean':
        return np.mean(corr_allinstances)