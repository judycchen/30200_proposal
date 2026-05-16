# Code by Shan Gao


import numpy as np
import pandas as pd
from scipy.stats import pearsonr



def loo_isc(sample_by_region_3d):

    """
    compute leave-one-out (one-to-others) isc
    input: 3d matrix (subj * sample * region), each layer is 1 subject's data (timepoints * region)
    output: 2d isc matrix (subj * region)
    """
    
    # print('Computing ISC...')

    sample_by_region_3d = np.array(sample_by_region_3d)
    
    num_subj = sample_by_region_3d.shape[0]
    num_region = sample_by_region_3d.shape[2]
    
    # create empty matrix for isc to be filled in: subj * region
    subj_by_region_isc_2d = np.zeros((num_subj, num_region))
    
    # compute isc
    for i in range(num_subj):  # for each layer (subj)
        
        # print('subj', i)
        
        # get the data layer for current subj
        curr_subj = sample_by_region_3d[i, :, :]
        
        # get all other subjs' data layers
        others_indices = np.setdiff1d(np.arange(num_subj), i)
        others_data = sample_by_region_3d[others_indices, :, :]
        
        # get timepoints * region mean values for all other subjs except subj i
        others_mean = np.nanmean(others_data, axis=0)
        
        # compute isc by region: 
        # corr between curr_subj and others_mean across all timepoints
        for region in range(num_region):
            subj_by_region_isc_2d[i, region] = pearsonr(curr_subj[:, region], others_mean[:, region])[0]
    
    return subj_by_region_isc_2d



def pairwise_isc(sample_by_region_3d):

    """
    compute pairwise isc
    input: 3d matrix (subj * sample * region), each layer is 1 subject's data (timepoints * region)
    output: 2d isc matrix (subj * region)
    """
    
    # print('Computing ISC...')

    # conver to np array
    sample_by_region_3d = np.array(sample_by_region_3d)
    
    # get dim
    num_subj = sample_by_region_3d.shape[0]
    num_region = sample_by_region_3d.shape[2]

    # create empty matrix for isc to be filled in: subj * region
    subj_by_region_isc_2d = np.zeros((num_subj, num_region))

    # compute isc
    for region in range(num_region):    # for each sagittal slice (region)

        # print('region', i)

        # slice out all subjs' data for the curr region (subj * timepoints), transpose (timepoints * subj)
        curr_region = sample_by_region_3d[:, :, region].T
        # convert to df for corr computation
        curr_region = pd.DataFrame(curr_region)
        # pairwise subj corr for curr_region
        pairwise_subj_corr = curr_region.corr()    # subj * subj symmetrical df

        for subj in range(num_subj):
            # get an array of corrs with everyone else
            pairwise_subj_corr_currsubj = pairwise_subj_corr.iloc[:, subj][pairwise_subj_corr.index != subj] # exclude corr with self
            subj_by_region_isc_2d[subj, region] = np.nanmean(pairwise_subj_corr_currsubj)

    return subj_by_region_isc_2d



def average_isc(subj_by_region_2d, ax=0):
# TODO: ah-oh i wrote a same helper function in mean_corr lol

    """
    average across rows/cols of subj_by_region_2d matrix to compute per region(axis=0)/subj(axis=1) mean isc
    (perform fisher's z transform before averaging, and back-transform to corr after averaging)
    input: 2d isc matrix (subj * region)
    output: 
    """

    # fisher's z transform
    z_isc_2d = np.arctanh(subj_by_region_2d)

    # average
    averaged_z_isc_1d = np.nanmean(z_isc_2d, axis=ax)

    # back-transform from fisher's z to corr
    averaged_isc_1d = np.tanh(averaged_z_isc_1d)

    return averaged_isc_1d