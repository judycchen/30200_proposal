"""
m_task_glm_contrast.py
----------------------
Aim 1: Does fNIRS→fMRI translation preserve task-evoked activation?

Two parallel GLM analyses on the N-back session:

  Analysis A — Predicted fMRI (122 ROIs, from l_apply_model_tasks.py)
    Per-subject GLM: [reg_1back | reg_2back | intercept]
    Contrast: 2-back > 1-back  →  second-level one-sample t-test across subjects
    Output: t-map in ROI space, painted onto atlas NIfTI, rendered with nilearn

  Analysis B — Raw fNIRS (42 channels, from b_fnirs-preproc/5_excluded)
    Same design matrices; GLM run per channel instead of per ROI
    Output: t-value per channel, bar plot with significance markers

Comparison between A and B: within Analysis A, after FDR thresholding, the
matching ROI for each fNIRS channel (from c_fnirs-fmri-corr/channel_roi_matches.csv)
is used to produce a channel-aligned scatter / side-by-side bar comparison.

GLM design (per subject):
  Columns  : [reg_1back, reg_2back, intercept]
  Regressors: boxcar convolved with SPM double-gamma HRF (evaluated at TR = 1 s)
  Contrast  : [-1, +1, 0]  →  2-back > 1-back

Block timing from k_parse_timing_behavior.py (nback_block_timing.json).
Predicted fMRI from l_apply_model_tasks.py (Nback_pred_SxRxT.mat).
Raw fNIRS from data/b_fnirs-preproc/5_excluded/all_Nback_zhbo_TxRxS.mat.

Inputs:
  - data/l_task-pred-fmri/Nback_pred_SxRxT.mat    (n_subj, 122, 300)
  - data/k_task-timing/nback_block_timing.json
  - data/b_fnirs-preproc/5_excluded/all_Nback_zhbo_TxRxS.mat  (300, 42, n_subj)
  - data/a_fmri-roi-ts/atlas_122_resampled.nii.gz
  - data/c_fnirs-fmri-corr/channel_roi_matches.csv  (optional; for A-B comparison)

Outputs (data/m_task-glm/):
  A_nback_beta_SxRx2.mat      : per-subject beta maps (n_subj, 122, 2)
  A_nback_contrast_SxR.mat    : per-subject 2back−1back contrast (n_subj, 122)
  A_nback_tstat_Rx1.mat       : group t-statistic per ROI (122,)
  A_nback_pval_fdr_Rx1.mat    : FDR-corrected p-values (122,)
  A_nback_contrast_tmasked.nii.gz  : FDR-thresholded t-map in volume space

  B_nback_beta_SxCx2.mat      : per-subject beta maps (n_subj, 42, 2)
  B_nback_contrast_SxC.mat    : per-subject contrast per channel (n_subj, 42)
  B_nback_tstat_Cx1.mat       : group t-statistic per channel (42,)
  B_nback_pval_fdr_Cx1.mat    : FDR-corrected p-values (42,)

  results/m_task-glm/:
  A_nback_brain_tstat.png     : nilearn glass-brain render
  A_nback_brain_tstat_slices.png : nilearn slice render
  B_nback_fnirs_contrast_bar.png : bar plot per fNIRS channel
  AB_comparison_bar.png       : A vs B side-by-side at matched ROIs

Usage:
  python scripts/m_task_glm_contrast.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma
from statsmodels.stats.multitest import multipletests

HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)

np.random.seed(86)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = "/project/ycleong/users/judycchen/prediction-proj"
PRED_FILE    = os.path.join(BASE, "data/l_task-pred-fmri/Nback_pred_SxRxT.mat")
TIMING_FILE  = os.path.join(BASE, "data/k_task-timing/nback_block_timing.json")
FNIRS_FILE   = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_Nback_zhbo_TxRxS.mat")
ATLAS_FILE   = os.path.join(BASE, "data/a_fmri-roi-ts/atlas_122_resampled.nii.gz")
CHAN_ROI_CSV = os.path.join(BASE, "data/c_fnirs-fmri-corr/channel_roi_matches.csv")
OUT_DIR      = os.path.join(BASE, "data/m_task-glm")
FIG_DIR      = os.path.join(BASE, "results/m_task-glm")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────
TR        = 1.0
N_TRS     = 300
N_ROIS    = 122
N_CH      = 42
FDR_Q     = 0.05
CONTRAST  = np.array([-1.0, 1.0, 0.0])   # 2-back > 1-back, ignore intercept

# ── helpers ───────────────────────────────────────────────────────────────────

def make_hrf(tr, duration=32.0):
    """SPM canonical double-gamma HRF evaluated at TR resolution."""
    t = np.arange(0, duration, tr)
    hrf = (t**5 * np.exp(-t)) / gamma(6) - (t**15 * np.exp(-t)) / (6 * gamma(16))
    hrf /= hrf.max()
    return hrf

def make_design_matrix(blocks, n_trs, tr, hrf):
    """
    blocks : list of dicts with keys onset_TR, offset_TR, condition
    Returns dm (n_trs, 3) with columns [reg_1back, reg_2back, intercept].
    """
    dm = np.zeros((n_trs, 3))
    dm[:, 2] = 1.0   # intercept

    for b in blocks:
        onset  = max(0, int(b['onset_TR']))
        offset = min(n_trs, int(b['offset_TR']))
        if onset >= offset:
            continue
        boxcar = np.zeros(n_trs)
        boxcar[onset:offset] = 1.0
        convolved = np.convolve(boxcar, hrf)[:n_trs]
        col = 0 if b['condition'] == '1back' else 1
        dm[:, col] += convolved

    return dm

def fit_glm(signal_T, dm_TxP, contrast_P):
    """
    OLS GLM fit for a single time series.
    Returns: beta (P,), contrast_val (scalar), t_stat (scalar)
    """
    n, p = dm_TxP.shape
    beta, _, _, _ = np.linalg.lstsq(dm_TxP, signal_T, rcond=None)
    contrast_val  = float(contrast_P @ beta)
    residuals     = signal_T - dm_TxP @ beta
    sigma2        = np.sum(residuals**2) / max(n - p, 1)
    XtX_inv       = np.linalg.pinv(dm_TxP.T @ dm_TxP)
    se            = np.sqrt(sigma2 * float(contrast_P @ XtX_inv @ contrast_P))
    t_stat        = contrast_val / se if se > 1e-12 else 0.0
    return beta, contrast_val, t_stat

def paint_rois(tstat_Rx1, atlas_img, roi_labels):
    """Paint per-ROI t-statistics onto the atlas volume."""
    atlas_data = atlas_img.get_fdata().astype(int)
    stat_vol   = np.zeros(atlas_data.shape, dtype=np.float32)
    for i, roi_id in enumerate(roi_labels):
        stat_vol[atlas_data == roi_id] = tstat_Rx1[i]
    return nib.Nifti1Image(stat_vol, atlas_img.affine, atlas_img.header)

# ── load inputs ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading inputs")
print("=" * 60)

with open(TIMING_FILE) as f:
    timing_all = json.load(f)                          # {subj_id: [blocks]}

pred_mat   = sio.loadmat(PRED_FILE)
pred_SxRxT = pred_mat['pred_SxRxT']                   # (n_subj, 122, 300)
print(f"Predicted fMRI: {pred_SxRxT.shape}  (subjects × ROIs × TRs)")

# load subject IDs saved by l_apply_model_tasks.py
SUBJ_IDS_FILE = os.path.join(BASE, "data/l_task-pred-fmri/Nback_subj_ids.json")
if os.path.exists(SUBJ_IDS_FILE):
    with open(SUBJ_IDS_FILE) as f:
        pred_subj_ids = json.load(f)
else:
    pred_subj_ids = [f's{i+1:02d}' for i in range(pred_SxRxT.shape[0])]
    print("WARNING: Nback_subj_ids.json not found — run l_ first")
print(f"Predicted fMRI subject IDs: {pred_subj_ids}")

fnirs_mat     = sio.loadmat(FNIRS_FILE)
fnirs_raw     = fnirs_mat['zhbo_TxRxS']                # (300, 42, n_subj)
# extract fNIRS subject IDs from excluded mat (same file b2 produced)
if 'subj_ids_included' in fnirs_mat:
    raw_ids       = fnirs_mat['subj_ids_included'].flatten()
    fnirs_subj_ids = [str(x.flat[0]) if hasattr(x, 'flat') else str(x) for x in raw_ids]
else:
    fnirs_subj_ids = pred_subj_ids
print(f"Raw fNIRS     : {fnirs_raw.shape}  (TRs × channels × subjects)")
print(f"fNIRS subject IDs: {fnirs_subj_ids}")

# intersect: subjects with predicted fMRI AND timing AND raw fNIRS
common_ids = [sid for sid in pred_subj_ids if sid in timing_all and sid in fnirs_subj_ids]
print(f"Subjects with all data: {len(common_ids)}  {common_ids}")

pred_idx  = {sid: i for i, sid in enumerate(pred_subj_ids)}
fnirs_idx = {sid: i for i, sid in enumerate(fnirs_subj_ids)}
n_subj    = len(common_ids)
print(f"Subjects for GLM: {n_subj}")

atlas_img  = nib.load(ATLAS_FILE)
atlas_data = atlas_img.get_fdata().astype(int)
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels > 0]
print(f"Atlas ROIs: {len(roi_labels)}")

hrf = make_hrf(TR)
print(f"HRF duration: {len(hrf) * TR:.0f}s  (evaluated at TR={TR}s)")

# ── ANALYSIS A: Predicted fMRI GLM ───────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS A: Predicted fMRI GLM (2-back > 1-back)")
print("=" * 60)

A_beta_SxRx2   = np.full((n_subj, N_ROIS, 2), np.nan)
A_contrast_SxR = np.full((n_subj, N_ROIS),    np.nan)

for s, subj_id in enumerate(common_ids):
    blocks = timing_all[subj_id]
    dm     = make_design_matrix(blocks, N_TRS, TR, hrf)   # (300, 3)

    pred_RxT = pred_SxRxT[pred_idx[subj_id]]   # (122, 300)

    for roi in range(N_ROIS):
        beta, contrast_val, _ = fit_glm(pred_RxT[roi], dm, CONTRAST)
        A_beta_SxRx2[s, roi, :]  = beta[:2]
        A_contrast_SxR[s, roi]   = contrast_val

    if (s + 1) % 5 == 0 or s == n_subj - 1:
        print(f"  A: subject {s+1}/{n_subj} done")

# group-level one-sample t-test
A_tstat_Rx1, A_pval_Rx1 = stats.ttest_1samp(A_contrast_SxR, popmean=0, axis=0)
A_reject_Rx1, A_pval_fdr_Rx1, _, _ = multipletests(A_pval_Rx1, alpha=FDR_Q, method='fdr_bh')
n_sig_pos_A = int(np.sum(A_reject_Rx1 & (A_tstat_Rx1 > 0)))
n_sig_neg_A = int(np.sum(A_reject_Rx1 & (A_tstat_Rx1 < 0)))
print(f"\nA: Significant ROIs (FDR q<{FDR_Q}): {A_reject_Rx1.sum()} / {N_ROIS}")
print(f"   2-back > 1-back: {n_sig_pos_A}  |  1-back > 2-back: {n_sig_neg_A}")

# save
sio.savemat(os.path.join(OUT_DIR, 'A_nback_beta_SxRx2.mat'),
            {'beta_SxRx2': A_beta_SxRx2, 'subj_ids': common_ids})
sio.savemat(os.path.join(OUT_DIR, 'A_nback_contrast_SxR.mat'),
            {'contrast_SxR': A_contrast_SxR})
sio.savemat(os.path.join(OUT_DIR, 'A_nback_tstat_Rx1.mat'),
            {'tstat_Rx1': A_tstat_Rx1, 'pval_Rx1': A_pval_Rx1,
             'pval_fdr_Rx1': A_pval_fdr_Rx1, 'reject_Rx1': A_reject_Rx1.astype(int)})
print("Saved A_nback_*.mat")

# brain volume: FDR-masked t-map
A_tstat_masked = A_tstat_Rx1.copy()
A_tstat_masked[~A_reject_Rx1] = 0.0
stat_img = paint_rois(A_tstat_masked, atlas_img, roi_labels)
nib.save(stat_img, os.path.join(OUT_DIR, 'A_nback_contrast_tmasked.nii.gz'))
print("Saved A_nback_contrast_tmasked.nii.gz")

# nilearn render
try:
    from nilearn import plotting

    # glass brain (4 views)
    fig_glass = plt.figure(figsize=(14, 4))
    display = plotting.plot_glass_brain(
        stat_img,
        colorbar=True,
        display_mode='lyrz',
        title=f'2-back > 1-back | predicted fMRI | FDR q<{FDR_Q}',
        figure=fig_glass,
        vmax=np.nanmax(np.abs(A_tstat_Rx1)),
        symmetric_cbar=True,
        cmap='RdBu_r',
    )
    fig_glass.savefig(os.path.join(FIG_DIR, 'A_nback_brain_tstat.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_glass)
    print("Saved A_nback_brain_tstat.png")

    # axial slices
    fig_slices = plt.figure(figsize=(14, 6))
    display2 = plotting.plot_stat_map(
        stat_img,
        display_mode='z',
        cut_coords=8,
        colorbar=True,
        title=f'2-back > 1-back | predicted fMRI | FDR q<{FDR_Q}',
        figure=fig_slices,
        vmax=np.nanmax(np.abs(A_tstat_Rx1)),
        symmetric_cbar=True,
        cmap='RdBu_r',
    )
    fig_slices.savefig(os.path.join(FIG_DIR, 'A_nback_brain_tstat_slices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_slices)
    print("Saved A_nback_brain_tstat_slices.png")

except ImportError:
    print("nilearn not available — skipping brain renders")

# ── ANALYSIS B: Raw fNIRS GLM ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS B: Raw fNIRS GLM (2-back > 1-back, per channel)")
print("=" * 60)

B_beta_SxCx2   = np.full((n_subj, N_CH, 2), np.nan)
B_contrast_SxC = np.full((n_subj, N_CH),    np.nan)

for s, subj_id in enumerate(common_ids):
    blocks   = timing_all[subj_id]
    dm       = make_design_matrix(blocks, N_TRS, TR, hrf)   # (300, 3)
    fnirs_TC = fnirs_raw[:, :, fnirs_idx[subj_id]]           # (300, 42)

    # impute NaNs channel-wise
    col_means = np.nanmean(fnirs_TC, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for c in range(fnirs_TC.shape[1]):
        nan_mask = np.isnan(fnirs_TC[:, c])
        fnirs_TC[nan_mask, c] = col_means[c]

    for ch in range(N_CH):
        beta, contrast_val, _ = fit_glm(fnirs_TC[:, ch], dm, CONTRAST)
        B_beta_SxCx2[s, ch, :]  = beta[:2]
        B_contrast_SxC[s, ch]   = contrast_val

    if (s + 1) % 5 == 0 or s == n_subj - 1:
        print(f"  B: subject {s+1}/{n_subj} done")

# group-level t-test
B_tstat_Cx1, B_pval_Cx1 = stats.ttest_1samp(B_contrast_SxC, popmean=0, axis=0)
B_reject_Cx1, B_pval_fdr_Cx1, _, _ = multipletests(B_pval_Cx1, alpha=FDR_Q, method='fdr_bh')
n_sig_pos_B = int(np.sum(B_reject_Cx1 & (B_tstat_Cx1 > 0)))
n_sig_neg_B = int(np.sum(B_reject_Cx1 & (B_tstat_Cx1 < 0)))
print(f"\nB: Significant channels (FDR q<{FDR_Q}): {B_reject_Cx1.sum()} / {N_CH}")
print(f"   2-back > 1-back: {n_sig_pos_B}  |  1-back > 2-back: {n_sig_neg_B}")

# save
sio.savemat(os.path.join(OUT_DIR, 'B_nback_beta_SxCx2.mat'),
            {'beta_SxCx2': B_beta_SxCx2})
sio.savemat(os.path.join(OUT_DIR, 'B_nback_contrast_SxC.mat'),
            {'contrast_SxC': B_contrast_SxC})
sio.savemat(os.path.join(OUT_DIR, 'B_nback_tstat_Cx1.mat'),
            {'tstat_Cx1': B_tstat_Cx1, 'pval_Cx1': B_pval_Cx1,
             'pval_fdr_Cx1': B_pval_fdr_Cx1, 'reject_Cx1': B_reject_Cx1.astype(int)})
print("Saved B_nback_*.mat")

# fNIRS bar plot
fig_B, ax_B = plt.subplots(figsize=(14, 5))
colors_B = ['#b22222' if s else '#7a7a7a' for s in B_reject_Cx1]
ax_B.bar(np.arange(1, N_CH + 1), B_tstat_Cx1, color=colors_B, width=0.7)
ax_B.axhline(0, color='black', linewidth=0.8)
ax_B.set_xlabel('fNIRS channel', fontsize=12)
ax_B.set_ylabel('Group t-statistic (2-back > 1-back)', fontsize=12)
ax_B.set_title(
    f'Analysis B: Raw fNIRS contrast (2-back > 1-back)\n'
    f'red=significant (FDR q<{FDR_Q}), n={B_reject_Cx1.sum()}/{N_CH}',
    fontsize=11
)
ax_B.set_xticks(np.arange(1, N_CH + 1))
ax_B.set_xticklabels(np.arange(1, N_CH + 1), fontsize=7)
for ch in range(N_CH):
    if B_pval_fdr_Cx1[ch] < 0.01:
        ax_B.text(ch+1, B_tstat_Cx1[ch] + 0.1, '**', ha='center', va='bottom', fontsize=8)
    elif B_reject_Cx1[ch]:
        ax_B.text(ch+1, B_tstat_Cx1[ch] + 0.1, '*',  ha='center', va='bottom', fontsize=8)
plt.tight_layout()
fig_B.savefig(os.path.join(FIG_DIR, 'B_nback_fnirs_contrast_bar.png'), dpi=150)
plt.close(fig_B)
print("Saved B_nback_fnirs_contrast_bar.png")

# ── A vs B comparison at matched ROIs ────────────────────────────────────────
print("\n" + "=" * 60)
print("A vs B: comparison at channel-matched ROIs")
print("=" * 60)

if os.path.exists(CHAN_ROI_CSV):
    chan_roi_df     = pd.read_csv(CHAN_ROI_CSV)
    matched_roi_ids = chan_roi_df['matched_roi'].values    # ROI label per channel (1-indexed label)

    # map ROI label → index in tstat array (roi_labels is sorted unique labels)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(roi_labels)}

    A_tstat_matched = np.array([
        A_tstat_Rx1[label_to_idx[int(roi_id)]] if int(roi_id) in label_to_idx else np.nan
        for roi_id in matched_roi_ids
    ])   # (42,) — predicted fMRI t-stat at each channel's matched ROI

    fig_AB, ax_AB = plt.subplots(figsize=(14, 5))
    x     = np.arange(1, N_CH + 1)
    width = 0.35
    bars_A = ax_AB.bar(x - width/2, A_tstat_matched, width, label='Predicted fMRI (matched ROI)',
                       color='#2166ac', alpha=0.8)
    bars_B = ax_AB.bar(x + width/2, B_tstat_Cx1,     width, label='Raw fNIRS',
                       color='#d6604d', alpha=0.8)
    ax_AB.axhline(0, color='black', linewidth=0.8)
    ax_AB.set_xlabel('fNIRS channel / matched ROI', fontsize=12)
    ax_AB.set_ylabel('Group t-statistic (2-back > 1-back)', fontsize=12)
    ax_AB.set_title('A vs B: Predicted fMRI (at matched ROI) vs Raw fNIRS', fontsize=11)
    ax_AB.set_xticks(x)
    ax_AB.set_xticklabels(x, fontsize=7)
    ax_AB.legend(fontsize=10)
    plt.tight_layout()
    fig_AB.savefig(os.path.join(FIG_DIR, 'AB_comparison_bar.png'), dpi=150)
    plt.close(fig_AB)
    print("Saved AB_comparison_bar.png")

    # correlation between A and B at matched ROIs
    valid = np.isfinite(A_tstat_matched) & np.isfinite(B_tstat_Cx1)
    if valid.sum() > 2:
        r_AB, p_AB = stats.pearsonr(A_tstat_matched[valid], B_tstat_Cx1[valid])
        print(f"Correlation between A (matched ROI) and B (fNIRS): r={r_AB:.3f}, p={p_AB:.4f}")
        sio.savemat(os.path.join(OUT_DIR, 'AB_comparison.mat'),
                    {'A_tstat_matched': A_tstat_matched, 'B_tstat_Cx1': B_tstat_Cx1,
                     'r_AB': r_AB, 'p_AB': p_AB})
else:
    print(f"channel_roi_matches.csv not found at {CHAN_ROI_CSV} — skipping A-B comparison")

print(f"\nAll outputs in: {OUT_DIR}  |  Figures in: {FIG_DIR}")
print("Done: m_task_glm_contrast.py")
