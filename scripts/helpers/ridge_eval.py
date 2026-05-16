# Code by Shan Gao


import numpy as np
from sklearn.linear_model import Ridge

import sys
from rowwise_pearsonr import rowwise_pearsonr_one_to_one



def ridge_pred(X_train_TxR, y_train_TxR,
               alpha,
               X_test_TxR):
    
    ridge = Ridge(alpha)
    ridge.fit(X_train_TxR, y_train_TxR)
    ridge_weights = ridge.coef_    # R_y x R_X
    y_test_pred_TxR = ridge.predict(X_test_TxR)   # OR: X_test_TxR @ ridge_weights.T

    return y_test_pred_TxR, ridge_weights



def ridge_eval(X_train_TxR, y_train_TxR,
               alpha,
               X_test_TxR, y_test_TxR,
               matrices=0):
    
    y_test_pred_TxR, ridge_weights = \
        ridge_pred(X_train_TxR, y_train_TxR,
                    alpha,
                    X_test_TxR)
    
    # evaluation: corr
    if matrices == 0:

        corr_per_target = rowwise_pearsonr_one_to_one(y_test_pred_TxR.T, y_test_TxR.T)

        return y_test_pred_TxR, corr_per_target, ridge_weights

    elif matrices == 1:

        corr_matrix_pred_true_by_pred_true = np.corrcoef(y_test_pred_TxR.T, y_test_TxR.T)   # 2num_targets * 2num_targets
        # get submatrices
        size = y_test_TxR.shape[1]  # number of targets
        corr_matrix_pred_by_pred = corr_matrix_pred_true_by_pred_true[:size, :size]  # top left quarter
        corr_matrix_true_by_pred = corr_matrix_pred_true_by_pred_true[size:, :size]   # bottom left quarter
        corr_matrix_true_by_true = corr_matrix_pred_true_by_pred_true[size:, size:]   # bottom right quarter
        corr_per_target = np.diagonal(corr_matrix_true_by_pred)

        return y_test_pred_TxR, corr_matrix_pred_by_pred, corr_matrix_true_by_pred, corr_matrix_true_by_true, corr_per_target, ridge_weights


    