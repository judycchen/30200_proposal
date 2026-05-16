## Code by Shan Gao


from scipy.stats import pearsonr, spearmanr
import numpy as np


def corr_mtx(mtx1, mtx2, metric='spearman'):

    """ 
    compute pearson's correlation between two matrices
    inputs: two matrices (np array), must have the same amount of non-nan values
    output: pearson's r (scalar)
    """

    if metric == 'spearman':
        return spearmanr(mtx1[~np.isnan(mtx1)].flatten(), mtx2[~np.isnan(mtx2)].flatten())[0]
    elif metric == 'pearson':
        return pearsonr(mtx1[~np.isnan(mtx1)].flatten(), mtx2[~np.isnan(mtx2)].flatten())[0]