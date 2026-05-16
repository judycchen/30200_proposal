# Code by Shan Gao


import numpy as np
import mean_corr



def loo_isfc(data_SxRxT):

    """ 
    in: time course SxRxT
    out: 
    - ISFC by subj: SxRxR 
    - mean ISFC across subj: RxR
    """

    nS = data_SxRxT.shape[0]
    nR = data_SxRxT.shape[1]

    isfc_SxRxR = []
    for subj in range(nS):

        # get the data layer for current subj
        currS_RxT = data_SxRxT[subj, :, :]

        # get all other subjs' data layers and average
        othersubj = np.setdiff1d(np.arange(nS), subj)
        meanS_RxT = np.nanmean(data_SxRxT[othersubj, :, :], axis=0)

        # compute corr matrix (currS by meanS) with NaN handling
        corr_mtx_currmeanxcurrmean = np.ma.corrcoef(np.ma.masked_invalid(currS_RxT), np.ma.masked_invalid(meanS_RxT))  # 2nR x 2nR
        isfc_SxRxR.append(corr_mtx_currmeanxcurrmean[:nR, nR:])   # top right quarter (currS by meanS)

    # convert to np array — use .filled(np.nan) so masked positions become NaN,
    # not the masked array's fill_value (which is often 0 or 1e+20)
    isfc_SxRxR = np.array([a.filled(np.nan) if hasattr(a, 'filled') else np.array(a)
                            for a in isfc_SxRxR])

    # average across subj
    isfc_RxR = mean_corr.mean_corr(isfc_SxRxR, axis=0)

    return isfc_SxRxR, isfc_RxR