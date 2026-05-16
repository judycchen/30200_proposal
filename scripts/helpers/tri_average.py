# Code by Shan Gao


import numpy as np
import mean_corr


def tri_average(data_RxR):

    """ 
    Given an ISFC matrix, average upper and lower triangles: (i, j) & (j, i)
    output: averaged ISFC filled in lower triangle (upper triangle are nans)
    """

    nR = data_RxR.shape[0]

    # initialize an array with nan for averaged values to be filled in
    data_averaged_RxR = np.full((nR, nR), np.nan)
    
    # fill in averaged values
    for i in range(nR):
        for j in range(i, nR):

            if i == j:   # diagonal
                data_averaged_RxR[i, j] = data_RxR[i, j]
            else:   # off-diagonal
                data_averaged_RxR[j, i] = mean_corr.mean_corr(
                    np.array([data_RxR[i, j], data_RxR[j, i]])
                )

    return np.array(data_averaged_RxR)