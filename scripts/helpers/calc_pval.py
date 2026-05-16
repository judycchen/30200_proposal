# Code by Shan Gao

import numpy as np
from statsmodels.stats.multitest import multipletests



def calc_pval(null_distrib, true_value, tail=1):

    """ 
    inputs:
    - null_distrib: 1d array of null distrib
    - true_value: scalar
    - tail: -1 (left-tailed), 0 (one-tailed either side), 1 (right-tailed), 2 (two-tailed).
    output: p value (scalar)
    """    

    count_all = len(null_distrib) + 1

    # left-tailed
    if tail == -1:
        count_greater = np.sum(null_distrib <= true_value) + 1

    # right-tailed
    elif tail == 1:
        count_greater = np.sum(null_distrib >= true_value) + 1

    # one-tailed either side
    elif tail == 0:

        if true_value >= 0:
            count_greater = np.sum(null_distrib >= true_value) + 1
        else:
            count_greater = np.sum(null_distrib <= true_value) + 1

    # two-tailed
    elif tail == 2:

        count_greater = np.sum(np.abs(null_distrib) >= np.abs(true_value)) + 1
    
    return count_greater / count_all



def fdr_pmask(d, var_name, fdr_method, fdr_alpha):

    """ 
    mutate dictionary d to append pval_corrected, sigmask, sigroi, pmasked var_name.
    input d should contain:
        - d[var_name+'_Rx1']: storing a 1d array of stat values (e.g., corr)
        - d[var_name+'_pval_Rx1']: storing a 1d array of uncorrected pvals for each stat value.
    """

    # raw pval
    pval_uncorrected = np.squeeze(d[var_name+'_pval_Rx1'])

    # get significance mask for rois, corrected pval
    sigmask, pval_corrected, _, _ = multipletests(pval_uncorrected, fdr_alpha, fdr_method)
    d[var_name+'_pval_corrected_Rx1'] = pval_corrected
    d[var_name+'_sigmask_Rx1'] = sigmask

    # sanity check fdr correction
    print(f'{var_name}: {sum((pval_corrected - pval_uncorrected) < 0)} ROI has corrected pval greater than raw pval.')

    # get a list of significant rois
    d[var_name+'_sigroi'] = np.where(pval_corrected < 0.05)[0] + 1

    # mask var_name array: set non-sig roi's corr to 0
    d[var_name+'_pmasked_Rx1'] = np.squeeze(d[var_name+'_Rx1']) * sigmask



def pval2mark(pval):

    if pval < 0.001:
        return "***"
    
    elif pval < 0.01:
        return "**"
    
    elif pval < 0.05:
        return "*"
    
    return "n.s."