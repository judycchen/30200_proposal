# Code by Shan Gao


import numpy as np
import pandas as pd
import os



def get_yeo_brainnetome(list_outputs, 
                        networks_all_order = ['Vis', 'SomMot', 'DorsAttn','SalVentAttn', 'Limbic', 'Cont', 'DefaultTP', 'Sub']):


    """ 
    inputs:
        - list_outputs: a list of variable names (str) to be output. choose from
            -- 'trueid_by_network_UL': dict with key:value pairs of network:roi_trueid, unilateral
            -- 'trueid_by_network_BL': dict with key:value pairs of network:roi_trueid, bilateral
            -- 'trueid_all_order_BL': vector with roi_trueid arranged according to networks_all_order, bilateral (within each network, all LH rois precedes RH rois)
            -- 'network_seg_LH': vector with idx for network segmentation line position, left hemisphere
            -- 'network_seg_RH': vector with idx for network segmentation line position, right hemisphere
            -- 'network_seg_BL': vector with idx for network segmentation line position, bilateral
        - networks_all_order: a list of network names (str) specifying network ordering. 'trueid_all_order_BL', 'network_seg_LH', 'network_seg_RH', 'network_seg_BL' to be sorted according to this order.
    usage: 
        get_yeo_brainnetome(['trueid_by_network_UL', 'trueid_by_network_BL', 'trueid_all_order_BL', 'network_seg_LH', 'network_seg_RH', 'network_seg_BL'])
    """


    # load roi info
    dirname = os.path.dirname(__file__)     # get path for current file

    print(dirname)

    roi_info = pd.read_csv('/project/ycleong/users/judycchen/fnirs_fmri_models/data/sherlock/fmri/masks/Yeo_17N_114_Brainnetome_subcortical_8/Yeo_17N_114_Brainnetome_subcortical_8.csv',
        header=None,
        names=['trueid', 'name_long', 'name_short', 'num1', 'num2', 'num3', 'num4']
                          )    
    # roi_info = pd.read_csv(
    #     os.path.join(dirname, '../../data/fmri/masks/Yeo_17N_114_Brainnetome_subcortical_8/Yeo_17N_114_Brainnetome_subcortical_8.csv'),     # relative path
    #     header=None, 
    #     names=['trueid', 'name_long', 'name_short', 'num1', 'num2', 'num3', 'num4']
    # )


    # initialize trueid_by_network_UL, trueid_by_network_BL
    trueid_by_network_UL = {
        'LVis': [],
        'LSomMot': [],
        'LDorsAttn': [],
        'LSalVentAttn': [],
        'LLimbic': [],
        'LCont': [],
        'LDefaultTP': [],
        'LSub': [],
        'RVis': [],
        'RSomMot': [],
        'RDorsAttn': [],
        'RSalVentAttn': [],
        'RLimbic': [],
        'RCont': [],
        'RDefaultTP': [],
        'RSub': []
    }
    trueid_by_network_BL = {
        'Vis': [],
        'SomMot': [],
        'DorsAttn': [],
        'SalVentAttn': [],
        'Limbic': [],
        'Cont': [],
        'DefaultTP': [],
        'Sub': []
    }
    

    # fill in trueid_by_network_UL, trueid_by_network_BL
    for idx, roi in roi_info.iterrows():

        if 'LH_Vis' in roi[1]:
            trueid_by_network_UL['LVis'].append(roi[0])
            trueid_by_network_BL['Vis'].append(roi[0])
        elif 'LH_SomMot' in roi[1]:
            trueid_by_network_UL['LSomMot'].append(roi[0])
            trueid_by_network_BL['SomMot'].append(roi[0])
        elif 'LH_DorsAttn' in roi[1]:
            trueid_by_network_UL['LDorsAttn'].append(roi[0])
            trueid_by_network_BL['DorsAttn'].append(roi[0])
        elif 'LH_SalVentAttn' in roi[1]:
            trueid_by_network_UL['LSalVentAttn'].append(roi[0])
            trueid_by_network_BL['SalVentAttn'].append(roi[0])
        elif 'LH_Limbic' in roi[1]:
            trueid_by_network_UL['LLimbic'].append(roi[0])
            trueid_by_network_BL['Limbic'].append(roi[0])
        elif 'LH_Cont' in roi[1]:
            trueid_by_network_UL['LCont'].append(roi[0])
            trueid_by_network_BL['Cont'].append(roi[0])
        elif 'LH_Default' in roi[1] or 'LH_TempPar' in roi[1]:
            trueid_by_network_UL['LDefaultTP'].append(roi[0])
            trueid_by_network_BL['DefaultTP'].append(roi[0])
        elif 'LH_Subcortical' in roi[1]:
            trueid_by_network_UL['LSub'].append(roi[0])
            trueid_by_network_BL['Sub'].append(roi[0])
        elif 'RH_Vis' in roi[1]:
            trueid_by_network_UL['RVis'].append(roi[0])
            trueid_by_network_BL['Vis'].append(roi[0])
        elif 'RH_SomMot' in roi[1]:
            trueid_by_network_UL['RSomMot'].append(roi[0])
            trueid_by_network_BL['SomMot'].append(roi[0])
        elif 'RH_DorsAttn' in roi[1]:
            trueid_by_network_UL['RDorsAttn'].append(roi[0])
            trueid_by_network_BL['DorsAttn'].append(roi[0])
        elif 'RH_SalVentAttn' in roi[1]:
            trueid_by_network_UL['RSalVentAttn'].append(roi[0])
            trueid_by_network_BL['SalVentAttn'].append(roi[0])
        elif 'RH_Limbic' in roi[1]:
            trueid_by_network_UL['RLimbic'].append(roi[0])
            trueid_by_network_BL['Limbic'].append(roi[0])
        elif 'RH_Cont' in roi[1]:
            trueid_by_network_UL['RCont'].append(roi[0])
            trueid_by_network_BL['Cont'].append(roi[0])
        elif 'RH_Default' in roi[1] or 'RH_TempPar' in roi[1]:
            trueid_by_network_UL['RDefaultTP'].append(roi[0])
            trueid_by_network_BL['DefaultTP'].append(roi[0])
        elif 'RH_Subcortical' in roi[1]:
            trueid_by_network_UL['RSub'].append(roi[0])
            trueid_by_network_BL['Sub'].append(roi[0])


    # get trueid_all_order_BL
    if 'trueid_all_order_BL' in list_outputs:
        trueid_all_order_BL = []
        for network in networks_all_order:
            trueid_all_order_BL += trueid_by_network_BL[network]
        trueid_all_order_BL = np.array(trueid_all_order_BL)

    
    # get left hemi network seg idx
    if 'network_seg_LH' in list_outputs:
        network_seg_LH = []
        curr = 0
        for network in networks_all_order[:-1]:
            curr += len(trueid_by_network_UL['L'+network])
            network_seg_LH.append(curr) 
        network_seg_LH = np.array(network_seg_LH)
    

    # get right hemi network seg idx
    if 'network_seg_RH' in list_outputs:
        network_seg_RH = []
        curr = 0
        for network in networks_all_order[:-1]:
            curr += len(trueid_by_network_UL['R'+network])
            network_seg_RH.append(curr) 
        network_seg_RH = np.array(network_seg_RH)
    

    # get bilateral network seg idx
    if 'network_seg_BL' in list_outputs:
        network_seg_BL = []
        curr = 0
        for network in networks_all_order[:-1]:
            curr += len(trueid_by_network_BL[network])
            network_seg_BL.append(curr) 
        network_seg_BL = np.array(network_seg_BL)


    # convert list to np.array in trueid_by_network_UL, trueid_by_network_BL for easier future conversion to idx if needed
    # must be done AFTER getting ordering vector
    for network, rois in trueid_by_network_UL.items():
        trueid_by_network_UL[network] = np.array(rois)
    for network, rois in trueid_by_network_BL.items():
        trueid_by_network_BL[network] = np.array(rois)


    # initialize dict for storing outputs
    outputs = {}

    # fill in outputs dict
    for output in list_outputs:
        outputs[output] = eval(output)

    # output
    return outputs



def get_nirx_pfc8x8(var_to_return):

    """ 
    var_to_return: any one from ['idx_all_order_BL', 'trueid_all_order_BL', 'trueid_by_network_BL', 'network_seg_idx_BL]
    """

    trueid_by_network_BL = {
        'DL': [2, 17, 1, 18],
        'VL': [3, 20, 4, 19],
        'DM': [5, 15, 8, 10, 7, 14, 9],
        'VM': [6, 16, 11, 13, 12]
    }

    trueid_all_order_BL = [
        2, 17, 1, 18,
        3, 20, 4, 19,
        5, 15, 8, 10, 7, 14, 9,
        6, 16, 11, 13, 12
    ]

    idx_all_order_BL = np.array(trueid_all_order_BL) - 1

    network_seg_idx_BL = [4,    # DL (up to 4, same below)
                          8,    # VL
                          15    # DM
                          ]

    return eval(var_to_return)