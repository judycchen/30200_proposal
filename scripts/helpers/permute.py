# Code by Shan Gao


import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from nltools.stats import phase_randomize

import sys
from rowwise_pearsonr import *
from calc_pval import *



def permute_one_to_manyregions(TC_permute_T, TC_compare_RxT, n_iters=1000, true_corr=None):

    """ 
    permute 1 TC (TC_permute_T), compare to many TCs (TC_compare_RxT) to generate **a pval for each of them** (e.g., compare an fmri control region to all fnirs channels)
    inputs:
        - TC_permute_T: 1d array of length T; to be permuted
        - TC_compare_RxT: 1d array of shape RxT; each row is a TC obs to be compared to TC_permute_T
        - n_iters: how many permutations to run
        - true_corr: true corr between TC_permute_T and each obs in TC_compare_RxT
    """

    d = {}

    # if true_corr is not provided, compute true_corr first
    if true_corr is None:
        d['corr_Rx1'] = rowwise_pearsonr_many_to_one(TC_compare_RxT, TC_permute_T, metric=None)
    else:
        d['corr_Rx1'] = true_corr
        
    # permutate to compute null corr for each obs in TC_compare_RxT
    null_corr_NxR = []
    for i in range(n_iters):
        if i % 100 == 0: print(f'iter {i+1}')
        null_TC_permute_T = phase_randomize(TC_permute_T, random_state=None)
        null_corr_R = rowwise_pearsonr_many_to_one(TC_compare_RxT, null_TC_permute_T, metric=None)
        null_corr_NxR.append(null_corr_R)
    d['null_corr_RxN'] = np.array(null_corr_NxR).T

    # compute pval for each obs
    d['corr_pval_Rx1'] = [
        calc_pval(d['null_corr_RxN'][i], d['corr_Rx1'][i], tail=1) for i in range(TC_compare_RxT.shape[0])
    ]

    # fdr correction across obs
    fdr_pmask(d, 'corr', 'fdr_bh', 0.05)

    return d



def phase_rand_many_to_one_singleregion(pred_tc_subjbytime, obsv_tc, metric='median', n_iters=1000, tail=1):

    """ 
    inputs:
    - pred_tc_subjbytime: 2d array; each row is a predicted time course for the region
    - obsv_tc: 1d array; observed time course for the region
    - metric: take median or mean across subjs to get the region's summary corr
    - n_iters
    - tail: 1 (one-tailed); 2 (two-tailed)
    outputs: p_val, true_corr, null_distrib
    """

    ## true corr between predicted and observed time course for the region
    true_corr = rowwise_pearsonr_many_to_one(pred_tc_subjbytime, obsv_tc, metric)
    
    ## compute null corrs
    null_distrib = []

    for _ in range(n_iters):

        # phase rand
        permuted_obsv_tc = phase_randomize(obsv_tc, random_state=None)

        # median of all subj's corr with permuted_true_tc as one value in null distrib
        null_distrib.append(   
            rowwise_pearsonr_many_to_one(pred_tc_subjbytime, permuted_obsv_tc, metric)
        )
    
    ## calc p val
    p_val = calc_pval(null_distrib, true_corr, tail)

    return p_val, true_corr, null_distrib



def circle_shift_one_to_one_allregions(data_permute, data_ground_truth, n_iter=1000, tail=0):
## TODO: not functioning rn. fix bug in corr computation.


    '''
    input:
        - data_permute: the df to be permuted. obs * features
        - data_ground_truth: ground truth df to compute corr against for each permute. obs * features, same shape as data_permute
        - n_iter: number of iterations to run permutation.
        - tail: -1 (left-tail), 0 (two-tail), +1 (right-tailed). default = 0.
    output: 
        - true_r: true corr beween unpermuted data_permute and data_ground_truth. 1 value for each feature.
        - null_r: null corr between each permuted data_permute and data_ground_truth. feature * n_iter.
        - p_val: 1 value for each feature.
        - p_val_corrected: FDR corrected p val. 1 value for each feature.
        - reject: true for hypothesis that can be rejected for given alpha. 1 bool value for each feature.
    '''


    ## compute true corr: np 1d array of len # features
    true_r = np.array(data_permute.corrwith(data_ground_truth, axis=0))


    ## compute null corr

    # set up
    n_obs, n_feature = data_ground_truth.shape
    # n_iter = (n_obs - 1) // stepsize    # exclude 1 original position
    # permuted = data_permute.copy()      # create deep copy
    null_r = []

    # circle shift
    for i in range(n_iter):

        # generate random stepsize (1: n_obs) for current iter
        stepsize = random.randint(1, n_obs)

        # update permuted df: put the first stepsize rows to the end
        permuted = pd.concat([data_permute[stepsize:], data_permute[:stepsize]], axis=0)

        ## TODO: fix corr calculation: corrwith computes corr between elements with matched index
        # compute null corr: 1d np array of len # features
        curr_null_r = np.array(permuted.corrwith(data_ground_truth, axis=0))
        # append to null_r: n_iter * feature
        null_r.append(curr_null_r)

    # convert list to np 2d array and transpose (for easier slicing by feature): feature * n_iter
    null_r = np.array(null_r).T


    ## compute p values
    p_val = []

    for i in range(n_feature):

        # get data
        feature_i_true_r = true_r[i]
        feature_i_null_r = null_r[i]

        # 2-tail
        if tail == 0:
            big_null_r = np.logical_or(feature_i_null_r >= abs(feature_i_true_r), feature_i_null_r <= -abs(feature_i_true_r))  # 1d bool array
        # right-tail
        elif tail == 1:
            big_null_r = (feature_i_null_r >= feature_i_true_r)  # 1d bool array
        # left-tail
        elif tail == -1:
            big_null_r = (feature_i_null_r <= feature_i_true_r)  # 1d bool array
        
        # compute p value for curr feature
        p_val.append((big_null_r.sum() + 1) / n_iter)
    

    ## FDR correction of p val
    reject, p_val_corrected, _, _ = multipletests(p_val, alpha=0.05, method='fdr_bh')  # same as fdrcorrection(p_val)[1]


    return true_r, null_r, p_val, p_val_corrected, reject



