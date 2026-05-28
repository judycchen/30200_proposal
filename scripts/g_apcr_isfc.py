"""
g_apcr_isfc.py
--------------
Phase 4: ISFC validation of aPCR model predictions

Asks: does the predicted fMRI preserve large-scale connectivity structure?
Computes ISFC for both true and predicted fMRI, then compares the two
matrices using Spearman correlation (Mantel-style test).

Steps:
  1. Compute LOO-ISFC on true fMRI (59 subjects)
  2. Compute LOO-ISFC on predicted fMRI (15 LOO subjects)
  3. Triangle-average both matrices (average upper and lower triangles)
  4. Remove diagonal
  5. Spearman correlation between flattened true and predicted ISFC
  6. Permutation test: shuffle ROI labels 1000 times, recompute similarity
  7. Save ISFC matrices + stats

Inputs:
  - data/a_fmri-roi-ts/bold_TxRxS.npy          : (886, 122, 59) true fMRI
  - data/d_apcr-model/nnw_pred_SxRxT.mat        : (15, 122, 886) predicted fMRI

Outputs (data/d_apcr-model/):
  - apcr_isfc.mat   : true and predicted ISFC matrices + similarity stats

Usage:
  python scripts/g_apcr_isfc.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr

HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)

from isfc import loo_isfc
from tri_average import tri_average
from calc_pval import calc_pval

np.random.seed(86)

# ── variant flag ──────────────────────────────────────────────────────────────
# Must match the FRONTAL_ONLY setting used in e_apcr_model.py.

FRONTAL_ONLY  = False
NEW_SUBJ_ONLY = True   # must match e_apcr_model.py

# ── paths ─────────────────────────────────────────────────────────────────────

BASE      = "/project/ycleong/users/judycchen/prediction-proj"
_out_suffix = ("_frontal" if FRONTAL_ONLY else "") + ("_newsubj" if NEW_SUBJ_ONLY else "")
MODEL_DIR = os.path.join(BASE, f"data/d_apcr-model{_out_suffix}_pca75")
FMRI_FILE = os.path.join(BASE, "data/a_fmri-roi-ts/bold_TxRxS.npy")

N_PERM    = 1000
N_ROI     = 122

# ── load ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading data")
print("=" * 60)

bold_TxRxS = np.load(FMRI_FILE)                                   # (886, 122, 59)
d_pred     = sio.loadmat(os.path.join(MODEL_DIR, "nnw_pred_SxRxT.mat"))
pred_SxRxT = d_pred['pred_SxRxT']                                  # (15, 122, 886)

nT, nROI, nS_fmri  = bold_TxRxS.shape
nS_fnirs            = pred_SxRxT.shape[0]
print(f"True fMRI shape     : {bold_TxRxS.shape}  (T x ROI x S)")
print(f"Predicted fMRI shape: {pred_SxRxT.shape}  (S x ROI x T)")

# ── reshape to SxRxT for loo_isfc ─────────────────────────────────────────────

true_SxRxT = bold_TxRxS.transpose(2, 1, 0)   # (59, 122, 886)
# pred_SxRxT already in (15, 122, 886)

# ── ISFC: true fMRI ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Computing ISFC — true fMRI")
print("=" * 60)

true_isfc_SxRxR, true_isfc_RxR = loo_isfc(true_SxRxT)
print(f"True ISFC per-subject shape: {true_isfc_SxRxR.shape}")
print(f"True ISFC mean shape       : {true_isfc_RxR.shape}")

# Triangle-average and remove diagonal
true_isfc_tril_RxR = tri_average(true_isfc_RxR)
np.fill_diagonal(true_isfc_tril_RxR, np.nan)
true_lower_nan = np.isnan(true_isfc_tril_RxR[np.tril_indices(N_ROI, k=-1)]).sum()
print(f"True ISFC (tri-avg, diag removed) non-nan: {(~np.isnan(true_isfc_tril_RxR)).sum()}")
print(f"True ISFC NaN in lower triangle (excl diag): {true_lower_nan} / {N_ROI*(N_ROI-1)//2}")

# ── ISFC: predicted fMRI ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Computing ISFC — predicted fMRI")
print("=" * 60)

pred_isfc_SxRxR, pred_isfc_RxR = loo_isfc(pred_SxRxT)
print(f"Pred ISFC per-subject shape: {pred_isfc_SxRxR.shape}")
print(f"Pred ISFC mean shape       : {pred_isfc_RxR.shape}")

pred_isfc_tril_RxR = tri_average(pred_isfc_RxR)
np.fill_diagonal(pred_isfc_tril_RxR, np.nan)
pred_lower_nan = np.isnan(pred_isfc_tril_RxR[np.tril_indices(N_ROI, k=-1)]).sum()
print(f"Pred ISFC NaN in lower triangle (excl diag): {pred_lower_nan} / {N_ROI*(N_ROI-1)//2}")

# ── matrix similarity ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Computing matrix similarity (Spearman)")
print("=" * 60)

# Flatten lower triangles (exclude diagonal and NaNs)
true_flat = true_isfc_tril_RxR[np.tril_indices(N_ROI, k=-1)]
pred_flat = pred_isfc_tril_RxR[np.tril_indices(N_ROI, k=-1)]

# Remove positions where either is NaN
valid = np.isfinite(true_flat) & np.isfinite(pred_flat)
true_flat = true_flat[valid]
pred_flat = pred_flat[valid]

true_sim, _ = pearsonr(true_flat, pred_flat)
print(f"True ISFC vs Predicted ISFC Pearson r = {true_sim:.4f}")

# ── permutation test: shuffle ROI labels ─────────────────────────────────────

print(f"\nPermutation test ({N_PERM} iters, shuffling ROI labels)...")
null_sim = []

for i in range(N_PERM):
    if (i + 1) % 200 == 0:
        print(f"  iter {i+1}/{N_PERM}...")

    # shuffle ROI order in predicted ISFC
    perm_idx = np.random.permutation(N_ROI)
    pred_perm = pred_isfc_tril_RxR[perm_idx, :][:, perm_idx]
    pred_perm_flat = pred_perm[np.tril_indices(N_ROI, k=-1)]

    valid_p = np.isfinite(true_flat) & np.isfinite(pred_perm_flat[valid])
    if valid_p.sum() < 10:
        null_sim.append(np.nan)
        continue

    r_perm, _ = pearsonr(true_flat[valid_p], pred_perm_flat[valid][valid_p])
    null_sim.append(r_perm)

null_sim = np.array(null_sim)
isfc_pval = calc_pval(null_sim[np.isfinite(null_sim)], true_sim, tail=1)
print(f"Matrix similarity: r={true_sim:.4f}, p={isfc_pval:.4f}")

# ── diagnostic: predicted fMRI variance per ROI ───────────────────────────────
# If non-covered ROIs have near-zero predicted variance, the 42-channel
# coverage hypothesis explains the white gaps (genuine near-zero ISFC, not NaN).

print("\n" + "=" * 60)
print("DIAGNOSTIC: predicted fMRI variance per ROI")
print("=" * 60)

pred_std_per_roi = pred_SxRxT.std(axis=(0, 2))   # (122,) std across subjects & time
print(f"Predicted fMRI std per ROI — mean: {pred_std_per_roi.mean():.4f}, "
      f"min: {pred_std_per_roi.min():.4f}, max: {pred_std_per_roi.max():.4f}")
print(f"ROIs with std < 0.01: {(pred_std_per_roi < 0.01).sum()}")
print(f"ROIs with std < 0.10: {(pred_std_per_roi < 0.10).sum()}")

# Also report per-ROI NaN count in the predicted lower triangle
pred_tril_i, pred_tril_j = np.tril_indices(N_ROI, k=-1)
pred_lower_vals = pred_isfc_tril_RxR[pred_tril_i, pred_tril_j]
nan_per_roi_pred = np.zeros(N_ROI, dtype=int)
for k in range(len(pred_tril_i)):
    if np.isnan(pred_lower_vals[k]):
        nan_per_roi_pred[pred_tril_i[k]] += 1
        nan_per_roi_pred[pred_tril_j[k]] += 1
top_nan_rois = np.argsort(nan_per_roi_pred)[::-1][:10]
if nan_per_roi_pred.max() > 0:
    print(f"\nTop 10 ROIs by NaN count in pred lower triangle (1-indexed):")
    for r in top_nan_rois:
        if nan_per_roi_pred[r] > 0:
            print(f"  ROI {r+1}: {nan_per_roi_pred[r]} NaN entries, pred std={pred_std_per_roi[r]:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

sio.savemat(os.path.join(MODEL_DIR, "apcr_isfc.mat"), {
    # true fMRI ISFC
    'true_isfc_SxRxR':      true_isfc_SxRxR,
    'true_isfc_RxR':        true_isfc_RxR,
    'true_isfc_tril_RxR':   true_isfc_tril_RxR,
    # predicted fMRI ISFC
    'pred_isfc_SxRxR':      pred_isfc_SxRxR,
    'pred_isfc_RxR':        pred_isfc_RxR,
    'pred_isfc_tril_RxR':   pred_isfc_tril_RxR,
    # similarity stats
    'isfc_similarity':      true_sim,
    'null_isfc_sim_N':      null_sim,
    'isfc_pval':            isfc_pval,
})

print(f"Saved: apcr_isfc.mat")
print(f"\nSummary:")
print(f"  True fMRI ISFC median  : {np.nanmedian(true_isfc_tril_RxR):.4f}")
print(f"  Pred fMRI ISFC median  : {np.nanmedian(pred_isfc_tril_RxR):.4f}")
print(f"  Matrix similarity      : r={true_sim:.4f}, p={isfc_pval:.4f}")
print(f"\nAll outputs in: {MODEL_DIR}")
print("Done: g_apcr_isfc.py")
