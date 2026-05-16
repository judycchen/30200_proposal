"""
c2_fnirs_fmri_corr_controls.py
--------------------------------
Phase 3 supplement: Control analysis — Supplementary Table S2

Correlates each fNIRS channel against three control fMRI time series:
  1. Primary auditory cortex  (ROIs 10 + 67 averaged — LH/RH SomMotB_Aud)
  2. Primary visual cortex    (ROIs 1-5 + 58-62 averaged — LH/RH visual)
  3. Global grey matter       (mean across all 122 ROIs)

Per Gao et al.: none of these should survive FDR correction, confirming
that the Phase 3 correlations are spatially specific rather than reflecting
global noise shared between modalities.

Inputs:
  - data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat  (fNIRS)
  - data/a_fmri-roi-ts/bold_groupmean_TxR.npy               (fMRI group mean)

Outputs (results/c_fnirs-fmri-corr/):
  - supp_table_s2_control_analysis.csv
  - fnirs_fmri_corr.mat  (updated with control results appended)

Usage:
  python scripts/c2_fnirs_fmri_corr_controls.py
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multitest import multipletests
from nltools.stats import phase_randomize

np.random.seed(86)

# ── paths ─────────────────────────────────────────────────────────────────────

BASE       = "/project/ycleong/users/judycchen/prediction-proj"
FNIRS_FILE = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat")
FMRI_MEAN  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
OUT_DIR    = os.path.join(BASE, "results/c_fnirs-fmri-corr")
MAT_FILE   = os.path.join(BASE, "data/c_fnirs-fmri-corr/fnirs_fmri_corr.mat")
os.makedirs(OUT_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────

N_PERM         = 1000
FDR_ALPHA      = 0.05
N_STANDARD_CH  = 42
DROP_FIRST_N   = 3

# ROI indices (0-based)
AUD_ROI_IDX = [9, 66]                          # ROIs 10, 67 — LH/RH SomMotB_Aud
VIS_ROI_IDX = [0, 1, 2, 3, 4, 57, 58, 59, 60, 61]  # ROIs 1-5, 58-62 — LH/RH visual

# ── load fNIRS group mean ─────────────────────────────────────────────────────

print("=" * 60)
print("Loading data")
print("=" * 60)

fnirs_mat    = sio.loadmat(FNIRS_FILE)
zhbo_TxRxS   = fnirs_mat['zhbo_TxRxS'][DROP_FIRST_N:, :, :]   # (886, 42, 15)
fnirs_mean   = np.nanmean(zhbo_TxRxS, axis=2)                  # (886, 42)
print(f"fNIRS group mean shape: {fnirs_mean.shape}")

fmri_mean    = np.load(FMRI_MEAN)                              # (886, 122)
print(f"fMRI group mean shape : {fmri_mean.shape}")

# ── build control time series ─────────────────────────────────────────────────

aud_ts    = fmri_mean[:, AUD_ROI_IDX].mean(axis=1)    # (886,)
vis_ts    = fmri_mean[:, VIS_ROI_IDX].mean(axis=1)    # (886,)
global_ts = fmri_mean.mean(axis=1)                    # (886,)

ctrl_names = ['primary_auditory', 'primary_visual', 'global_grey_matter']
ctrl_ts    = [aud_ts, vis_ts, global_ts]

# ── run control analysis ──────────────────────────────────────────────────────

ctrl_results = {}

for ctrl_name, ctrl_bold in zip(ctrl_names, ctrl_ts):

    print("\n" + "=" * 60)
    print(f"Control: {ctrl_name}")
    print("=" * 60)

    # Pearson r per channel
    ctrl_r = np.zeros(N_STANDARD_CH)
    for ch in range(N_STANDARD_CH):
        hbo_ts = fnirs_mean[:, ch]
        keep   = np.isfinite(hbo_ts) & np.isfinite(ctrl_bold)
        if keep.sum() < 3:
            ctrl_r[ch] = np.nan
        else:
            ctrl_r[ch], _ = stats.pearsonr(hbo_ts[keep], ctrl_bold[keep])

    print(f"  Median r = {np.nanmedian(ctrl_r):.4f}")

    # Phase-randomization permutation test
    pvals = np.zeros(N_STANDARD_CH)
    for ch in range(N_STANDARD_CH):
        if (ch + 1) % 10 == 0:
            print(f"  Permutation — channel {ch+1}/{N_STANDARD_CH}...")
        hbo_ts = fnirs_mean[:, ch]
        keep   = np.isfinite(hbo_ts) & np.isfinite(ctrl_bold)
        if keep.sum() < 3:
            pvals[ch] = np.nan
            continue
        nd = []
        for i in range(N_PERM):
            bold_perm = phase_randomize(ctrl_bold[keep], random_state=None)
            r_p, _    = stats.pearsonr(hbo_ts[keep], bold_perm)
            nd.append(r_p)
        pvals[ch] = (np.sum(np.array(nd) >= ctrl_r[ch]) + 1) / (N_PERM + 1)

    # FDR correction
    reject, qvals, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method='fdr_bh')
    n_sig = reject.sum()
    print(f"  Significant: {n_sig}/{N_STANDARD_CH} (FDR q<{FDR_ALPHA})")

    ctrl_results[ctrl_name] = {
        'r': ctrl_r, 'p': pvals, 'q': qvals,
        'significant': reject.astype(int),
        'median_r': np.nanmedian(ctrl_r),
        'n_sig': n_sig,
    }

# ── save Supp Table S2 CSV ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

rows = []
for ch in range(N_STANDARD_CH):
    row = {'channel': ch + 1}
    for ctrl_name in ctrl_names:
        row[f'{ctrl_name}_r']   = round(ctrl_results[ctrl_name]['r'][ch], 4)
        row[f'{ctrl_name}_q']   = round(ctrl_results[ctrl_name]['q'][ch], 4)
        row[f'{ctrl_name}_sig'] = ctrl_results[ctrl_name]['significant'][ch]
    rows.append(row)

supp_s2_df = pd.DataFrame(rows)
csv_path   = os.path.join(OUT_DIR, "supp_table_s2_control_analysis.csv")
supp_s2_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── append to main .mat ───────────────────────────────────────────────────────

if os.path.exists(MAT_FILE):
    d_main = sio.loadmat(MAT_FILE)
    for ctrl_name in ctrl_names:
        d_main[f'ctrl_{ctrl_name}_r']   = ctrl_results[ctrl_name]['r']
        d_main[f'ctrl_{ctrl_name}_q']   = ctrl_results[ctrl_name]['q']
        d_main[f'ctrl_{ctrl_name}_sig'] = ctrl_results[ctrl_name]['significant']
    sio.savemat(MAT_FILE, d_main)
    print(f"Control results appended to: {MAT_FILE}")
else:
    print(f"WARNING: {MAT_FILE} not found — saving controls as separate .mat")
    sio.savemat(os.path.join(OUT_DIR, "fnirs_fmri_corr_controls.mat"),
                {f'ctrl_{k}_{s}': ctrl_results[k][s]
                 for k in ctrl_names for s in ['r', 'q', 'significant']})

# ── print summary ─────────────────────────────────────────────────────────────

print("\nSupp Table S2 summary:")
print(f"{'Control':<25} {'Median r':>10} {'N sig':>8}")
print("-" * 45)
for ctrl_name in ctrl_names:
    print(f"{ctrl_name:<25} "
          f"{ctrl_results[ctrl_name]['median_r']:>10.4f} "
          f"{ctrl_results[ctrl_name]['n_sig']:>8}/{N_STANDARD_CH}")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done: c2_fnirs_fmri_corr_controls.py")
