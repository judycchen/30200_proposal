"""
e_apcr_model.py
---------------
Phase 4: aPCR predictive model — fNIRS → fMRI

Leave-one-fNIRS-subject-out cross-validation.
For each held-out fNIRS subject:
  - Train on remaining 14 fNIRS × 59 fMRI subjects (826 pairs)
  - Concatenate all pairs along time axis
  - PCA on fNIRS (90% variance) and fMRI (90% variance), matching Ryleigh/Gao pipeline
  - OLS regression: fNIRS PCs → fMRI PCs
  - Predict held-out subject's fMRI (886 × 122)
  - Back-project to ROI space

Evaluation:
  - indvdl-grp: per-subject predicted vs group-mean fMRI, median r across subjects per ROI
  - grp-grp:   mean-across-LOO predicted vs group-mean fMRI, r per ROI

Inputs:
  - data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat   (889, 42, 15)
  - data/a_fmri-roi-ts/bold_TxRxS.npy                        (886, 122, 59)
  - data/a_fmri-roi-ts/bold_groupmean_TxR.npy                (886, 122)

Outputs (data/d_apcr-model/):
  - nnw_pred_SxRxT.mat    : predicted fMRI per LOO subject (15, 122, 886)
  - nnw_pred_RxT.mat      : group-mean predicted fMRI (122, 886)
  - apcr_model_corr.mat   : indvdl-grp and grp-grp r per ROI

Usage:
  python scripts/e_apcr_model.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ── add helpers to path ───────────────────────────────────────────────────────
HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)

from pcr_eval import pcr_pred
from concat_obs import concat_obs
from impute_nan import impute_nan
from rowwise_pearsonr import rowwise_pearsonr_one_to_one
from mean_corr import mean_corr

np.random.seed(86)

# ── variant flag ──────────────────────────────────────────────────────────────
# Set FRONTAL_ONLY = True to restrict to 22 anterior channels (cortical y > -30mm),
# dropping bilateral temporal/parietal channels (ch 10-23, 42-49 in 1-based numbering).
# Outputs go to d_apcr-model_frontal/ instead of d_apcr-model/.

FRONTAL_ONLY = False

# 0-indexed positions within the 42-channel standard array that are frontal (y > -30mm)
FRONTAL_CH_IDX_42 = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23,
                     24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# ── paths ─────────────────────────────────────────────────────────────────────

BASE       = "/project/ycleong/users/judycchen/prediction-proj"
FNIRS_FILE = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat")
FMRI_FILE  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_TxRxS.npy")
FMRI_MEAN  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
_out_suffix = "_frontal" if FRONTAL_ONLY else ""
OUT_DIR    = os.path.join(BASE, f"data/d_apcr-model{_out_suffix}_pca75")
os.makedirs(OUT_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────

DROP_FIRST_N       = 3     # drop first 3 fNIRS timepoints to align with fMRI
FNIRS_VAR_THRESHOLD = 0.75  # ~12 fNIRS PCs, matching Gao et al. 2025 (their 90% → 12.8 PCs)
FMRI_VAR_THRESHOLD  = 0.85  # ~52 fMRI PCs, closest to Gao's 49 PCs given our data structure
MAX_FNIRS_PCS = 42    # updated below after optional frontal masking
MAX_FMRI_PCS  = 122   # upper bound = total ROIs (effectively no cap)

# ── load data ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading data")
print("=" * 60)

# fNIRS: load all channels, exclude 8 short-sep channels → (886, 42, nS)
# Short channels (0-indexed): [4,11,20,23,27,31,38,49] (1-indexed in MATLAB: [5,12,21,24,28,32,39,50])
SHORT_CH_IDX     = [4, 11, 20, 23, 27, 31, 38, 49]
standard_ch_mask = np.ones(50, dtype=bool)
standard_ch_mask[SHORT_CH_IDX] = False

fnirs_mat    = sio.loadmat(FNIRS_FILE)
zhbo_raw     = fnirs_mat['zhbo_TxRxS']                         # (T, 50, nS) or (T, 42, nS)
if zhbo_raw.shape[1] == 50:
    zhbo_TxRxS = zhbo_raw[DROP_FIRST_N:, standard_ch_mask, :]  # (886, 42, nS)
else:
    zhbo_TxRxS = zhbo_raw[DROP_FIRST_N:, :, :]                 # already 42 channels
if FRONTAL_ONLY:
    zhbo_TxRxS = zhbo_TxRxS[:, FRONTAL_CH_IDX_42, :]           # (886, 22, nS)
nT, nCh, nS_fnirs = zhbo_TxRxS.shape
MAX_FNIRS_PCS = nCh   # cap PCs at actual channel count
print(f"fNIRS shape (after drop): {zhbo_TxRxS.shape}  (T x channels x subjects)")

# fMRI: (886, 122, 59)
bold_TxRxS   = np.load(FMRI_FILE)                              # (886, 122, 59)
bold_mean    = np.load(FMRI_MEAN)                              # (886, 122)
nT_fmri, nROI, nS_fmri = bold_TxRxS.shape
print(f"fMRI shape             : {bold_TxRxS.shape}  (T x ROIs x subjects)")
print(f"fMRI group mean shape  : {bold_mean.shape}")

assert nT == nT_fmri, f"Timepoint mismatch: fNIRS={nT}, fMRI={nT_fmri}"
print(f"\nfNIRS subjects: {nS_fnirs} | fMRI subjects: {nS_fmri} | T: {nT} | ROIs: {nROI}")

# ── reshape to SxTxR convention for helpers ───────────────────────────────────
# concat_obs expects SxTxR

fnirs_SxTxR = zhbo_TxRxS.transpose(2, 0, 1)    # (15, 886, 42)
fmri_SxTxR  = bold_TxRxS.transpose(2, 0, 1)    # (59, 886, 122)

# ── NaN imputation and PC utilities ──────────────────────────────────────────

def fill_nan_colmean(X):
    """Replace NaNs with column mean (0.0 if entire column is NaN)."""
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X = X.copy()
    X[nan_rows, nan_cols] = col_means[nan_cols]
    return X

def n_pcs_for_variance(data_TxR, threshold, max_pcs):
    """Return PCs needed for threshold variance, hard-capped at max_pcs."""
    pca = PCA().fit(data_TxR)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return min(n, max_pcs)

# Estimate on full data — actual count will vary per LOO iteration
fnirs_full = fill_nan_colmean(fnirs_SxTxR.reshape(-1, nCh))
fmri_full  = fmri_SxTxR.reshape(-1, nROI)
n_fnirs_pcs_est = n_pcs_for_variance(fnirs_full, FNIRS_VAR_THRESHOLD, MAX_FNIRS_PCS)
n_fmri_pcs_est  = n_pcs_for_variance(fmri_full,  FMRI_VAR_THRESHOLD,  MAX_FMRI_PCS)
print(f"\nEstimated PCs (fNIRS {FNIRS_VAR_THRESHOLD*100:.0f}%, fMRI {FMRI_VAR_THRESHOLD*100:.0f}%): "
      f"fNIRS={n_fnirs_pcs_est} (max {MAX_FNIRS_PCS}), fMRI={n_fmri_pcs_est} (max {MAX_FMRI_PCS})")

# ── LOO loop ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"LOO cross-validation ({nS_fnirs} iterations)")
print("=" * 60)

pred_SxRxT = np.zeros((nS_fnirs, nROI, nT))   # predicted fMRI per held-out subject
n_pcs_log  = []

for loo_idx in range(nS_fnirs):

    print(f"\n--- LOO {loo_idx+1}/{nS_fnirs} ---")

    # ── split train / test ────────────────────────────────────────────────────
    train_idx    = [i for i in range(nS_fnirs) if i != loo_idx]
    fnirs_train  = fnirs_SxTxR[train_idx, :, :]   # (14, 886, 42)
    fnirs_test   = fnirs_SxTxR[loo_idx,   :, :]   # (886, 42)
    # fMRI: all 59 subjects used for training every iteration
    fmri_train   = fmri_SxTxR                      # (59, 886, 122)

    # ── impute NaNs in training fNIRS ────────────────────────────────────────
    fnirs_train_RxTxS = fnirs_train.transpose(2, 1, 0)   # (42, 886, 14)
    fnirs_train_RxTxS = impute_nan(fnirs_train_RxTxS)
    fnirs_train       = fnirs_train_RxTxS.transpose(2, 1, 0)   # back to (14, 886, 42)

    # ── impute NaNs in test fNIRS (fill bad channels with column mean ≈ 0) ───
    fnirs_test = fill_nan_colmean(fnirs_test)

    # ── concatenate all train pairs along time ────────────────────────────────
    # concat_obs pairs every fNIRS train subject with every fMRI subject
    # 14 × 59 = 826 pairs, each of length 886 → concatenated T = 826 × 886
    X_train_TxR, y_train_TxR = concat_obs(fnirs_train, fmri_train)
    print(f"  Train set: {X_train_TxR.shape} fNIRS, {y_train_TxR.shape} fMRI "
          f"({len(train_idx)} × {nS_fmri} = {len(train_idx)*nS_fmri} pairs)")

    # ── determine PCs for this iteration (85% var, hard-capped) ──────────────
    n_fnirs_pcs = n_pcs_for_variance(X_train_TxR, FNIRS_VAR_THRESHOLD, MAX_FNIRS_PCS)
    n_fmri_pcs  = n_pcs_for_variance(y_train_TxR, FMRI_VAR_THRESHOLD,  MAX_FMRI_PCS)
    n_pcs_log.append((n_fnirs_pcs, n_fmri_pcs))
    print(f"  PCs: fNIRS={n_fnirs_pcs} ({FNIRS_VAR_THRESHOLD*100:.0f}%, max {MAX_FNIRS_PCS}), "
          f"fMRI={n_fmri_pcs} ({FMRI_VAR_THRESHOLD*100:.0f}%, max {MAX_FMRI_PCS})")

    # ── fit model and predict ─────────────────────────────────────────────────
    y_pred_TxR, _, _, _ = pcr_pred(
        X_train_TxR, y_train_TxR,
        n_fnirs_pcs, n_fmri_pcs,
        fnirs_test
    )   # (886, 122)

    pred_SxRxT[loo_idx, :, :] = y_pred_TxR.T   # store as (122, 886)
    print(f"  Predicted shape: {y_pred_TxR.shape}")

n_pcs_arr = np.array(n_pcs_log)
print(f"\nPCs across LOO — fNIRS: mean={n_pcs_arr[:,0].mean():.1f} (max {MAX_FNIRS_PCS}), "
      f"fMRI: mean={n_pcs_arr[:,1].mean():.1f} (max {MAX_FMRI_PCS})")

# ── group-mean prediction ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Computing group-mean prediction and evaluating")
print("=" * 60)

# Mean predicted fMRI across LOO subjects: (122, 886)
pred_mean_RxT = np.mean(pred_SxRxT, axis=0)

# grp-grp: correlate mean-predicted vs group-mean true fMRI, per ROI
bold_mean_RxT      = bold_mean.T   # (122, 886)
grp_grp_corr_Rx1   = rowwise_pearsonr_one_to_one(pred_mean_RxT, bold_mean_RxT)
print(f"grp-grp: median r={np.nanmedian(grp_grp_corr_Rx1):.4f}, "
      f"mean r={np.nanmean(grp_grp_corr_Rx1):.4f}")

# indvdl-grp: per-subject predicted vs group-mean true fMRI, median r per ROI
indvdl_corr_SxR = np.zeros((nS_fnirs, nROI))
for s in range(nS_fnirs):
    indvdl_corr_SxR[s, :] = rowwise_pearsonr_one_to_one(
        pred_SxRxT[s, :, :], bold_mean_RxT
    )

indvdl_grp_corr_Rx1 = np.median(indvdl_corr_SxR, axis=0)
print(f"indvdl-grp: median r={np.nanmedian(indvdl_grp_corr_Rx1):.4f}, "
      f"mean r={np.nanmean(indvdl_grp_corr_Rx1):.4f}")

# ── save ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving outputs")
print("=" * 60)

sio.savemat(os.path.join(OUT_DIR, "nnw_pred_SxRxT.mat"), {
    "pred_SxRxT":   pred_SxRxT,    # (15, 122, 886)
    "n_pcs_fnirs":  n_pcs_arr[:, 0],
    "n_pcs_fmri":   n_pcs_arr[:, 1],
})

sio.savemat(os.path.join(OUT_DIR, "nnw_pred_RxT.mat"), {
    "pred_mean_RxT": pred_mean_RxT,  # (122, 886)
})

sio.savemat(os.path.join(OUT_DIR, "apcr_model_corr.mat"), {
    "grp_grp_corr_Rx1":    grp_grp_corr_Rx1,
    "indvdl_grp_corr_Rx1": indvdl_grp_corr_Rx1,
    "indvdl_corr_SxR":     indvdl_corr_SxR,
})

print(f"Saved nnw_pred_SxRxT.mat  : shape {pred_SxRxT.shape}")
print(f"Saved nnw_pred_RxT.mat    : shape {pred_mean_RxT.shape}")
print(f"Saved apcr_model_corr.mat")
print(f"\nAll outputs in: {OUT_DIR}")
print("Done: e_apcr_model.py")
