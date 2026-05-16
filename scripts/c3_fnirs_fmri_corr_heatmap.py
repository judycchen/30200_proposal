"""
c3_fnirs_fmri_corr_heatmap.py
------------------------------
Standalone script to generate Phase 3 correlation heatmap.
Replicates b_unimodal-bimodal_fc_matrix from old pipeline.

Shows each fNIRS channel's r value against:
  - Matching ROI (nearest centroid match)
  - Global grey matter signal (mean across all 122 ROIs)

Colormap: blue→white→red, -0.5 to 0.5
Significance markers (* q<0.05, ** q<0.01) on right side

Inputs:
  - data/c_fnirs-fmri-corr/fnirs_fmri_corr.mat
  - data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat
  - data/a_fmri-roi-ts/bold_groupmean_TxR.npy

Outputs:
  - data/c_fnirs-fmri-corr/fnirs_fmri_corr_heatmap.png

Usage:
  python scripts/c3_fnirs_fmri_corr_heatmap.py
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

np.random.seed(86)

# ── paths ─────────────────────────────────────────────────────────────────────

BASE       = "/project/ycleong/users/judycchen/prediction-proj"
CORR_FILE  = os.path.join(BASE, "data/c_fnirs-fmri-corr/fnirs_fmri_corr.mat")
FNIRS_FILE = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat")
FMRI_MEAN  = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
OUT_DIR    = os.path.join(BASE, "data/c_fnirs-fmri-corr")

N_STANDARD_CH = 42
DROP_FIRST_N  = 3

# ── load existing correlation results ─────────────────────────────────────────

print("Loading correlation results...")
d = sio.loadmat(CORR_FILE)

grp_grp_corr_Rx1          = d['grp_grp_corr_Rx1'].flatten()
grp_grp_corr_sigmask_Rx1  = d['grp_grp_corr_sigmask_Rx1'].flatten().astype(bool)
grp_grp_corr_pval_corrected_Rx1 = d['grp_grp_corr_pval_corrected_Rx1'].flatten()

print(f"Loaded: {len(grp_grp_corr_Rx1)} channels, {grp_grp_corr_sigmask_Rx1.sum()} significant")

# ── compute global grey matter r ──────────────────────────────────────────────

print("Computing global grey matter correlations...")
fnirs_mat  = sio.loadmat(FNIRS_FILE)
zhbo       = fnirs_mat['zhbo_TxRxS'][DROP_FIRST_N:, :, :]   # (886, 42, 15)
fnirs_mean = np.nanmean(zhbo, axis=2)                        # (886, 42)

fmri_mean  = np.load(FMRI_MEAN)                             # (886, 122)
global_ts  = fmri_mean.mean(axis=1)                         # (886,)

global_r = np.zeros(N_STANDARD_CH)
for ch in range(N_STANDARD_CH):
    hbo_ts = fnirs_mean[:, ch]
    keep   = np.isfinite(hbo_ts) & np.isfinite(global_ts)
    if keep.sum() < 3:
        global_r[ch] = np.nan
    else:
        global_r[ch], _ = stats.pearsonr(hbo_ts[keep], global_ts[keep])

print(f"Global grey matter median r: {np.nanmedian(global_r):.4f}")

# ── build heatmap matrix (N_CH x 2) ──────────────────────────────────────────

corr_matrix = np.column_stack([grp_grp_corr_Rx1, global_r])  # (42, 2)

# ── plot ──────────────────────────────────────────────────────────────────────

print("Generating heatmap...")

cmap_bwr = LinearSegmentedColormap.from_list(
    'bwr_gao',
    ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'],
    N=256
)

fig, ax = plt.subplots(figsize=(3, 12))

im = ax.imshow(corr_matrix, cmap=cmap_bwr, vmin=-0.5, vmax=0.5,
               aspect='auto', interpolation='none')

# x-axis at top
ax.set_xticks([0, 1])
ax.set_xticklabels(['Matching', 'Yeo Global'], fontsize=10,
                    rotation=45, ha='left', va='bottom')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_xlabel('fMRI ROI', fontsize=11)

# y-axis: channel numbers
ax.set_yticks(np.arange(N_STANDARD_CH))
ax.set_yticklabels(np.arange(1, N_STANDARD_CH + 1), fontsize=7)
ax.set_ylabel('fNIRS Channel', fontsize=11)

# significance markers (* q<0.05, ** q<0.01) on right
for ch in range(N_STANDARD_CH):
    if grp_grp_corr_pval_corrected_Rx1[ch] < 0.01:
        ax.text(1.6, ch, '**', ha='left', va='center', fontsize=7)
    elif grp_grp_corr_sigmask_Rx1[ch]:
        ax.text(1.6, ch, '*',  ha='left', va='center', fontsize=7)

# grid lines
ax.set_yticks(np.arange(-0.5, N_STANDARD_CH, 1), minor=True)
ax.yaxis.grid(True, which='minor', color='white', linewidth=0.5)
ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
ax.xaxis.grid(True, which='minor', color='white', linewidth=1)

# colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.25, pad=0.15)
cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fnirs_fmri_corr_heatmap.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
print("Done: c3_fnirs_fmri_corr_heatmap.py")
