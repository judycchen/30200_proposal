# Code by Shan Gao


import numpy as np


def concat_obs(X_SxTxR, y_SxTxR):

    X_TxR = []
    y_TxR = []

    for subj_X in X_SxTxR:
        for subj_y in y_SxTxR:
            X_TxR.append(subj_X)  # TxR
            y_TxR.append(subj_y)

    X_TxR = np.concatenate(X_TxR, axis=0)
    y_TxR = np.concatenate(y_TxR, axis=0)

    return X_TxR, y_TxR