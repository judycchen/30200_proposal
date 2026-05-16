# Code by Shan Gao

import numpy as np

def corr_mtx_reorder(corr_mtx_original, ordering):

    corr_mtx_reordered = []

    for next_row in ordering:
        row_original = corr_mtx_original[next_row]
        corr_mtx_reordered.append(row_original[ordering])

    return np.array(corr_mtx_reordered)