# Code by Shan Gao


import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import sys
from rowwise_pearsonr import rowwise_pearsonr_one_to_one



def pcr_pred(X_train_TxR, y_train_TxR,
             X_num_pcs, y_num_pcs,
             X_test_TxR):
    
    """ 
    inputs: 
    - num_pcs specifies number of PCs to keep
    - training and test datasets
    outputs:
    - 2d pca transformation matrices for X and y
    - 2d linear regression weight matrix
    - predicted TC
    """

    # pca for X
    X_pca = PCA(n_components=X_num_pcs)
    X_train_pc = X_pca.fit_transform(X_train_TxR)   # TxPC
    X_test_pc = X_pca.transform(X_test_TxR)    # TxPC
    # get pc loadings for X
    X_pca_loadings = X_pca.components_    # PC_XxR

    # pca for y
    y_pca = PCA(n_components=y_num_pcs)
    y_train_pc = y_pca.fit_transform(y_train_TxR)   # TxPC
    # get pc loadings for y
    y_pca_loadings = y_pca.components_    # PCxR

    # fit linear reg model on training set
    lr_reg = LinearRegression().fit(X=X_train_pc, y=y_train_pc)
    # get weights
    lr_weights = lr_reg.coef_   # target (fmri pc) * weights on predictors (fnirs pc)

    # predict on test set
    y_test_pc_pred = lr_reg.predict(X=X_test_pc)   # TxPC
    # back-project y_test_pc_pred to original space
    y_test_pred_TxR = y_pca.inverse_transform(y_test_pc_pred)   # TxR

    return y_test_pred_TxR, X_pca_loadings, y_pca_loadings, lr_weights




def pcr_eval(X_train_TxR, y_train_TxR,
             X_num_pcs, y_num_pcs,
             X_test_TxR, y_test_TxR,
             matrices=0):
    
    """ 
    inputs:
    - num_pcs specifies number of PCs to keep
    - training and test datasets
    - matrices: whether to return 2d corr matrices (0:no; 1: yes)
    output: 
    - *2d corr matrix: corr_matrix_pred_by_pred, corr_matrix_pred_by_true, corr_matrix_true_by_true
    - 1d corr array between y_test_pred and y_test at each corresponding target: corr_per_target
    - 2d pca transformation matrices for X and y
    - 2d linear regression weight matrix
    - predicted TC
    """

    y_test_pred_TxR, X_pca_loadings, y_pca_loadings, lr_weights = \
        pcr_pred(X_train_TxR, y_train_TxR,
                 X_num_pcs, y_num_pcs,
                 X_test_TxR)

    # evaluation: corr
    if matrices == 0:

        corr_per_target = rowwise_pearsonr_one_to_one(y_test_pred_TxR.T, y_test_TxR.T)

        return y_test_pred_TxR, corr_per_target, X_pca_loadings, y_pca_loadings, lr_weights

    elif matrices == 1:

        corr_matrix_pred_true_by_pred_true = np.corrcoef(y_test_pred_TxR.T, y_test_TxR.T)   # 2num_targets * 2num_targets
        # get submatrices
        size = y_test_TxR.shape[1]  # number of targets
        corr_matrix_pred_by_pred = corr_matrix_pred_true_by_pred_true[:size, :size]  # top left quarter
        corr_matrix_true_by_pred = corr_matrix_pred_true_by_pred_true[size:, :size]   # bottom left quarter
        corr_matrix_true_by_true = corr_matrix_pred_true_by_pred_true[size:, size:]   # bottom right quarter
        corr_per_target = np.diagonal(corr_matrix_true_by_pred)

        return y_test_pred_TxR, corr_matrix_pred_by_pred, corr_matrix_true_by_pred, corr_matrix_true_by_true, corr_per_target, X_pca_loadings, y_pca_loadings, lr_weights

