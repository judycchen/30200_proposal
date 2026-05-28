"""
j_train_final_model.py
----------------------
Train a single aPCR model on ALL NNW fNIRS subjects and save the fitted
weights explicitly, so they can be applied to new task fNIRS data.

Background: e_apcr_model.py runs leave-one-out cross-validation and discards
each fold's weights after prediction. This script trains one model on the
complete NNW dataset (all available fNIRS × 59 fMRI subjects) and serialises the
three fitted objects needed for forward prediction:
  - fNIRS PCA  (fNIRS channels → PCs)
  - fMRI PCA   (fMRI ROIs → PCs, used for inverse projection only)
  - OLS weights (fNIRS PCs → fMRI PCs)

Uses the same hyperparameters as the default aPCR run (90% fNIRS variance,
85% fMRI variance). Saves components as plain numpy arrays to avoid sklearn
version-mismatch issues when loading the model in other scripts.

Inputs:
  - data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat  (T, channels, n_subj)
  - data/a_fmri-roi-ts/bold_TxRxS.npy                       (886, 122, 59)

Outputs (data/j_final-model/):
  - apcr_final_model.npz  : numpy arrays for PCA + OLS
    keys: fnirs_mean, fnirs_components, fnirs_n_pcs,
          fmri_mean,  fmri_components,  fmri_n_pcs,
          lr_coef, lr_intercept

Usage:
  python scripts/j_train_final_model.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)
from concat_obs import concat_obs
from impute_nan import impute_nan
from rowwise_pearsonr import rowwise_pearsonr_one_to_one

np.random.seed(86)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = "/project/ycleong/users/judycchen/prediction-proj"
FNIRS_FILE = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat")
FMRI_FILE  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_TxRxS.npy")
FMRI_MEAN  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
OUT_DIR    = os.path.join(BASE, "data/j_final-model")
OUT_FILE   = os.path.join(OUT_DIR, "apcr_final_model.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────
SHORT_CH_IDX        = [4, 11, 20, 23, 27, 31, 38, 49]
DROP_FIRST_N        = 3
FNIRS_VAR_THRESHOLD = 0.90
FMRI_VAR_THRESHOLD  = 0.85

# ── helpers ───────────────────────────────────────────────────────────────────

def n_pcs_for_variance(data_TxR, threshold, max_pcs):
    pca = PCA().fit(data_TxR)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return min(n, max_pcs)

def fill_nan_colmean(X):
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X = X.copy()
    X[nan_rows, nan_cols] = col_means[nan_cols]
    return X

def apply_model(fnirs_TxC, fnirs_mean, fnirs_components, fmri_mean, fmri_components,
                lr_coef, lr_intercept):
    """Forward pass: fNIRS (T × C) → predicted fMRI (T × 122)."""
    X_centered = fnirs_TxC - fnirs_mean
    X_pc       = X_centered @ fnirs_components.T
    y_pc_pred  = X_pc @ lr_coef.T + lr_intercept
    y_pred     = y_pc_pred @ fmri_components + fmri_mean
    return y_pred

# ── load fNIRS ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data")
print("=" * 60)

standard_ch_mask = np.ones(50, dtype=bool)
standard_ch_mask[SHORT_CH_IDX] = False

fnirs_mat = sio.loadmat(FNIRS_FILE)
zhbo_raw  = fnirs_mat['zhbo_TxRxS']                              # (889, 50, 21)
if zhbo_raw.shape[1] == 50:
    zhbo = zhbo_raw[DROP_FIRST_N:, standard_ch_mask, :]           # (886, 42, 21)
else:
    zhbo = zhbo_raw[DROP_FIRST_N:, :, :]
nT, nCh, nS_fnirs = zhbo.shape
print(f"fNIRS shape (after drop): {zhbo.shape}  (T x channels x subjects)")

fnirs_SxTxR = zhbo.transpose(2, 0, 1)                            # (21, 886, 42)

# impute NaNs: impute_nan expects (R, T, S)
fnirs_RxTxS = fnirs_SxTxR.transpose(2, 1, 0)                    # (42, 886, 21)
fnirs_RxTxS = impute_nan(fnirs_RxTxS)
fnirs_SxTxR = fnirs_RxTxS.transpose(2, 1, 0)                    # (21, 886, 42)

# ── load fMRI ─────────────────────────────────────────────────────────────────
bold_TxRxS  = np.load(FMRI_FILE)                                 # (886, 122, 59)
bold_mean   = np.load(FMRI_MEAN)                                 # (886, 122)
nT_fmri, nROI, nS_fmri = bold_TxRxS.shape
fmri_SxTxR  = bold_TxRxS.transpose(2, 0, 1)                     # (59, 886, 122)
print(f"fMRI shape             : {bold_TxRxS.shape}  (T x ROIs x subjects)")
assert nT == nT_fmri, f"Timepoint mismatch: fNIRS={nT}, fMRI={nT_fmri}"

# ── concatenate all pairs ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Concatenating all subject pairs")
print("=" * 60)

X_train, y_train = concat_obs(fnirs_SxTxR, fmri_SxTxR)
print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  {nS_fnirs} fNIRS × {nS_fmri} fMRI = {nS_fnirs * nS_fmri} pairs × {nT} TRs")

# ── estimate PC counts ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Estimating PC counts")
print("=" * 60)

n_fnirs_pcs = n_pcs_for_variance(X_train, FNIRS_VAR_THRESHOLD, nCh)
n_fmri_pcs  = n_pcs_for_variance(y_train, FMRI_VAR_THRESHOLD,  nROI)
print(f"fNIRS PCs: {n_fnirs_pcs}  ({FNIRS_VAR_THRESHOLD*100:.0f}% variance, max {nCh})")
print(f"fMRI  PCs: {n_fmri_pcs}  ({FMRI_VAR_THRESHOLD*100:.0f}% variance, max {nROI})")

# ── fit PCA and OLS ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting PCA + OLS")
print("=" * 60)

fnirs_pca = PCA(n_components=n_fnirs_pcs).fit(X_train)
fmri_pca  = PCA(n_components=n_fmri_pcs).fit(y_train)

X_pc = fnirs_pca.transform(X_train)   # (rows, n_fnirs_pcs)
y_pc = fmri_pca.transform(y_train)    # (rows, n_fmri_pcs)

lr = LinearRegression().fit(X_pc, y_pc)
print(f"OLS fitted: coef shape={lr.coef_.shape}, intercept shape={lr.intercept_.shape}")

# ── sanity check ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Sanity check: forward pass on one NNW fNIRS subject")
print("=" * 60)

s_check  = np.random.randint(nS_fnirs)
pred_check = apply_model(
    fnirs_SxTxR[s_check],
    fnirs_pca.mean_, fnirs_pca.components_,
    fmri_pca.mean_,  fmri_pca.components_,
    lr.coef_, lr.intercept_
)   # (886, 122)

corr_check = rowwise_pearsonr_one_to_one(pred_check.T, bold_mean.T)
print(f"Subject {s_check}: median r={np.nanmedian(corr_check):.4f}, "
      f"mean r={np.nanmean(corr_check):.4f} (vs group-mean fMRI)")
print("(For reference: LOO e_apcr_model grp-grp r is typically ~0.3–0.5 per ROI)")

# ── save ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving model")
print("=" * 60)

np.savez(OUT_FILE,
         fnirs_mean=fnirs_pca.mean_,
         fnirs_components=fnirs_pca.components_,
         fnirs_n_pcs=n_fnirs_pcs,
         fmri_mean=fmri_pca.mean_,
         fmri_components=fmri_pca.components_,
         fmri_n_pcs=n_fmri_pcs,
         lr_coef=lr.coef_,
         lr_intercept=lr.intercept_)

print(f"Saved: {OUT_FILE}")
print(f"  fnirs_mean        : {fnirs_pca.mean_.shape}")
print(f"  fnirs_components  : {fnirs_pca.components_.shape}  ({n_fnirs_pcs} PCs × {nCh} channels)")
print(f"  fmri_mean         : {fmri_pca.mean_.shape}")
print(f"  fmri_components   : {fmri_pca.components_.shape}  ({n_fmri_pcs} PCs × {nROI} ROIs)")
print(f"  lr_coef           : {lr.coef_.shape}")
print(f"  lr_intercept      : {lr.intercept_.shape}")
print("\nDone: j_train_final_model.py")
