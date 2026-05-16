"""
f_apcr_significance.py
-----------------------
Phase 4: Significance testing + PNC for aPCR model

Inputs (from e_apcr_model.py):
  - data/d_apcr-model/nnw_pred_RxT.mat       : group-mean predicted fMRI (122, 886)
  - data/d_apcr-model/apcr_model_corr.mat    : grp-grp and indvdl-grp r per ROI

Additional inputs:
  - data/a_fmri-roi-ts/bold_groupmean_TxR.npy : (886, 122) true group-mean fMRI
  - data/a_fmri-roi-ts/bold_TxRxS.npy         : (886, 122, 59) per-subject fMRI for PNC

Steps:
  1. Phase-randomization permutation test on grp-grp r (1000 iters, nltools)
  2. FDR correction across 122 ROIs (fdr_bh, q<0.05)
  3. Compute Proportion of Noise Ceiling (PNC)
     - Cronbach's alpha from per-subject fMRI as noise ceiling estimate
     - PNC = model r / sqrt(alpha)
  4. Save full results as .mat + summary CSV

Outputs (data/d_apcr-model/):
  - apcr_significance.mat   : full results including null distributions
  - apcr_summary.csv        : per-ROI r, p, q, significant, PNC

Usage:
  python scripts/f_apcr_significance.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
import pandas as pd
from statsmodels.stats.multitest import multipletests
from nltools.stats import phase_randomize
from scipy.stats import pearsonr

HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)

from rowwise_pearsonr import rowwise_pearsonr_one_to_one
from calc_pval import calc_pval, fdr_pmask

np.random.seed(86)

# ── variant flag ──────────────────────────────────────────────────────────────
# Must match the FRONTAL_ONLY setting used in e_apcr_model.py.

FRONTAL_ONLY = False

# ── paths ─────────────────────────────────────────────────────────────────────

BASE      = "/project/ycleong/users/judycchen/prediction-proj"
_out_suffix = "_frontal" if FRONTAL_ONLY else ""
MODEL_DIR = os.path.join(BASE, f"data/d_apcr-model{_out_suffix}_pca75")
FMRI_MEAN = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
FMRI_FILE = os.path.join(BASE, "data/a_fmri-roi-ts/bold_TxRxS.npy")

N_PERM    = 1000
FDR_ALPHA = 0.05
N_ROI     = 122

# ── load ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading model outputs")
print("=" * 60)

d_corr      = sio.loadmat(os.path.join(MODEL_DIR, "apcr_model_corr.mat"))
d_pred      = sio.loadmat(os.path.join(MODEL_DIR, "nnw_pred_RxT.mat"))

grp_grp_corr_Rx1    = d_corr['grp_grp_corr_Rx1'].flatten()
indvdl_grp_corr_Rx1 = d_corr['indvdl_grp_corr_Rx1'].flatten()

pred_mean_RxT = d_pred['pred_mean_RxT']        # (122, 886)
bold_mean_TxR = np.load(FMRI_MEAN)             # (886, 122)
bold_mean_RxT = bold_mean_TxR.T                # (122, 886)
bold_TxRxS    = np.load(FMRI_FILE)             # (886, 122, 59)

print(f"grp-grp r shape    : {grp_grp_corr_Rx1.shape}")
print(f"Predicted shape    : {pred_mean_RxT.shape}")
print(f"True fMRI shape    : {bold_mean_RxT.shape}")
print(f"Per-subject fMRI   : {bold_TxRxS.shape}")

# ── permutation test ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Phase-randomization permutation test ({N_PERM} iters)")
print("=" * 60)

null_grp_grp_corr_RxN = np.zeros((N_ROI, N_PERM))
grp_grp_corr_pval_Rx1 = np.zeros(N_ROI)

for roi in range(N_ROI):
    if (roi + 1) % 20 == 0:
        print(f"  ROI {roi+1}/{N_ROI}...")

    pred_T = pred_mean_RxT[roi, :]    # (886,)
    true_T = bold_mean_RxT[roi, :]    # (886,)

    null_dist = []
    for i in range(N_PERM):
        true_perm = phase_randomize(true_T, random_state=None)
        r_perm, _ = pearsonr(pred_T, true_perm)
        null_dist.append(r_perm)

    null_grp_grp_corr_RxN[roi, :] = null_dist
    grp_grp_corr_pval_Rx1[roi]    = calc_pval(null_dist, grp_grp_corr_Rx1[roi], tail=1)

print(f"Permutation complete.")

# ── FDR correction ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FDR correction (fdr_bh)")
print("=" * 60)

d_results = {
    'grp_grp_corr_Rx1':    grp_grp_corr_Rx1.reshape(-1, 1),
    'grp_grp_corr_pval_Rx1': grp_grp_corr_pval_Rx1.reshape(-1, 1),
}
fdr_pmask(d_results, 'grp_grp_corr', 'fdr_bh', FDR_ALPHA)

n_sig = d_results['grp_grp_corr_sigmask_Rx1'].sum()
print(f"Significant ROIs (FDR q<{FDR_ALPHA}): {n_sig}/{N_ROI}")
print(f"Significant ROI indices: {d_results['grp_grp_corr_sigroi']}")
print(f"Median r (all ROIs):     {np.nanmedian(grp_grp_corr_Rx1):.4f}")
print(f"Median r (sig ROIs):     {np.nanmedian(grp_grp_corr_Rx1[d_results['grp_grp_corr_sigmask_Rx1'].flatten().astype(bool)]):.4f}" 
      if n_sig > 0 else "No significant ROIs.")

# ── PNC: Proportion of Noise Ceiling ─────────────────────────────────────────
# Cronbach's alpha from fMRI subjects = estimate of max achievable r
# alpha = (N / (N-1)) * (1 - sum(var_i) / var_total)
# PNC = model_r / sqrt(alpha)

print("\n" + "=" * 60)
print("Computing Proportion of Noise Ceiling (PNC)")
print("=" * 60)

nS_fmri = bold_TxRxS.shape[2]
pnc_Rx1 = np.zeros(N_ROI)

for roi in range(N_ROI):
    roi_data = bold_TxRxS[:, roi, :]   # (886, 59) — T x subjects

    # Cronbach's alpha across subjects
    item_vars  = np.var(roi_data, axis=0, ddof=1)    # variance per subject
    total_var  = np.var(roi_data.sum(axis=1), ddof=1) # variance of sum
    alpha      = (nS_fmri / (nS_fmri - 1)) * (1 - item_vars.sum() / total_var)
    alpha      = max(alpha, 0)   # clamp to 0 if negative (unreliable ROI)

    # PNC = model r / sqrt(alpha) — per Gao et al. / Jiahui et al. 2020
    noise_ceil = np.sqrt(alpha)
    if noise_ceil > 0:
        pnc_Rx1[roi] = grp_grp_corr_Rx1[roi] / noise_ceil
    else:
        pnc_Rx1[roi] = np.nan

print(f"Median PNC (all ROIs):  {np.nanmedian(pnc_Rx1):.4f}")
print(f"Mean PNC (all ROIs):    {np.nanmean(pnc_Rx1):.4f}")
if n_sig > 0:
    sig_mask = d_results['grp_grp_corr_sigmask_Rx1'].flatten().astype(bool)
    print(f"Median PNC (sig ROIs):  {np.nanmedian(pnc_Rx1[sig_mask]):.4f}")

# ── save ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

# Full .mat file
save_dict = {
    'grp_grp_corr_Rx1':              grp_grp_corr_Rx1,
    'indvdl_grp_corr_Rx1':           indvdl_grp_corr_Rx1,
    'null_grp_grp_corr_RxN':         null_grp_grp_corr_RxN,
    'grp_grp_corr_pval_Rx1':         grp_grp_corr_pval_Rx1,
    'grp_grp_corr_pval_corrected_Rx1': d_results['grp_grp_corr_pval_corrected_Rx1'].flatten(),
    'grp_grp_corr_sigmask_Rx1':      d_results['grp_grp_corr_sigmask_Rx1'].flatten().astype(int),
    'grp_grp_corr_sigroi':           d_results['grp_grp_corr_sigroi'],
    'grp_grp_corr_pmasked_Rx1':      d_results['grp_grp_corr_pmasked_Rx1'].flatten(),
    'pnc_Rx1':                       pnc_Rx1,
}
sio.savemat(os.path.join(MODEL_DIR, "apcr_significance.mat"), save_dict)
print(f"Saved: apcr_significance.mat")

# Summary CSV — per-ROI table
summary_df = pd.DataFrame({
    'roi':          np.arange(1, N_ROI + 1),
    'pearson_r':    grp_grp_corr_Rx1.round(4),
    'indvdl_r':     indvdl_grp_corr_Rx1.round(4),
    'p_perm':       grp_grp_corr_pval_Rx1.round(4),
    'q_fdr':        d_results['grp_grp_corr_pval_corrected_Rx1'].flatten().round(4),
    'significant':  d_results['grp_grp_corr_sigmask_Rx1'].flatten().astype(int),
    'pnc':          pnc_Rx1.round(4),
})
csv_path = os.path.join(MODEL_DIR, "apcr_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"Saved: apcr_summary.csv")
print(f"\nTop 10 ROIs by r:")
print(summary_df.nlargest(10, 'pearson_r')[['roi','pearson_r','q_fdr','significant','pnc']].to_string(index=False))

print(f"\nAll outputs in: {MODEL_DIR}")
print("Done: f_apcr_significance.py")
