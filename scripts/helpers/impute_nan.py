## Code by Shan Gao

import numpy as np


def impute_nan(data, metric='mean'):

    """ 
    inputs:
    - data: np 3d array of shape __region * time * subj__
    - metric: mean/median
    output:
    - data, with nan TCs imputed with mean/median TC of the same region across subj
    """

    data_imputed = []

    for curr_region in data:     # time * subj

        if metric == 'mean':
            curr_region_imputed = np.apply_along_axis(lambda x: np.nan_to_num(x, nan=np.nanmean(x)), axis=1, arr=curr_region)     # impute by row (timepoint)
        elif metric == 'median':
            curr_region_imputed = np.apply_along_axis(lambda x: np.nan_to_num(x, nan=np.nanmedian(x)), axis=1, arr=curr_region)
        else:
            break

        data_imputed.append(curr_region_imputed)
    
    return np.array(data_imputed)