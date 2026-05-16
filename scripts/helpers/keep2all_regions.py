# Code by Shan Gao


import numpy as np


def keep2all_regions(keep_values, keep_regions, all_regions):
    
    """ 
    input: 
    - 1d/2d array of scores for keep_regions: keep_values
    - 1d arrays of keep_regions and all_regions IDs
    output: 
    - 1d/2d array of scores for all_regions, with scores for keep_regions filled in & others remain 0: all_values
    """

    # idx of keep_regions in all_regions array
    keep_regions_idx = np.where(np.in1d(all_regions, keep_regions))[0]
    # number of all regions
    num_all_regions = len(all_regions)

    # 1d array values
    if len(keep_values.shape) == 1:
        # get all_values for all_regions
        all_values = np.zeros(num_all_regions)
        all_values[keep_regions_idx] = keep_values

    # 2d array of values
    elif len(keep_values.shape) == 2:
        # get all_values for all_regions
        all_values = np.zeros((num_all_regions, num_all_regions))
        col_idx, row_idx = np.meshgrid(keep_regions_idx, keep_regions_idx)
        all_values[row_idx, col_idx] = keep_values

    return all_values