"""
c_fnirs_fmri_corr.py
--------------------
Phase 3: fNIRS-fMRI one-to-one sanity check correlation

Steps:
  1. Load fNIRS HbO (TxRxS), drop first 3 timepoints to align with fMRI
  2. Compute group-mean fNIRS (T x 42)
  3. Load fMRI group-mean BOLD (T x 122)
  4. Read MNI coordinates for 42 standard channels from Standard_Channels.txt
     (comma-separated; col 6=distance_mm, cols 7-9=MNI xyz; rows where col6 > 15mm)
  5. Compute 122 ROI centroids from resampled atlas
  6. Match each channel to nearest ROI; flag collisions and dist > 20mm
  7. Pearson r for each matched channel-ROI pair
  8. Phase-randomization permutation test using nltools (1000 iterations)
  9. FDR correction (q < 0.05, fdr_bh)
  10. Save results as .mat (consistent with MATLAB pipeline)
  11. Bar plot of r per channel with significance markers
  12. Save two NIfTI files for FSLeyes spatial sanity check

Inputs:
  - data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat
  - data/a_fmri-roi-ts/bold_groupmean_TxR.npy
  - data/a_fmri-roi-ts/atlas_122_resampled.nii.gz
  - /project/ycleong/datasets/CogTasks-NNW_judy/Montage/Standard_Channels.txt

Outputs (all in data/c_fnirs-fmri-corr/):
  - fnirs_fmri_corr.mat              : all stats results
  - channel_roi_matches.csv          : channel-to-ROI mapping with distances
  - fnirs_fmri_corr_barplot.png      : bar plot of r per channel
  - fnirs_channels_spheres.nii.gz    : FSLeyes sanity check — channel locations
  - roi_matches_labeled.nii.gz       : FSLeyes sanity check — matched ROIs

FSLeyes sanity check (run locally after scp):
  fsleyes atlas_122_resampled.nii.gz roi_matches_labeled.nii.gz fnirs_channels_spheres.nii.gz
  Each channel sphere should sit above its matched ROI in the brain.

Usage:
  python scripts/c_fnirs_fmri_corr.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from nltools.stats import phase_randomize

np.random.seed(86)

# ─── VARIANT FLAG ─────────────────────────────────────────────────────────────
# Set FRONTAL_ONLY = True to restrict to 22 anterior channels (cortical y > -30mm),
# dropping bilateral temporal/parietal channels (ch 10-23, 42-49 in 1-based numbering).
# Outputs go to c_fnirs-fmri-corr_frontal/ instead of c_fnirs-fmri-corr/.

FRONTAL_ONLY = True

# 0-indexed positions within the 42-channel standard array that are frontal (y > -30mm)
FRONTAL_CH_IDX_42 = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23,
                     24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE         = "/project/ycleong/users/judycchen/prediction-proj"
FNIRS_FILE   = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded/all_NNW_zhbo_TxRxS.mat")
FMRI_MEAN    = os.path.join(BASE, "data/a_fmri-roi-ts/bold_groupmean_TxR.npy")
ATLAS_FILE   = os.path.join(BASE, "data/a_fmri-roi-ts/atlas_122_resampled.nii.gz")
STDCHAN_FILE = "/project/ycleong/datasets/CogTasks-NNW_judy/Montage/Standard_Channels.txt"
_out_suffix  = "_frontal" if FRONTAL_ONLY else ""
OUT_DIR      = os.path.join(BASE, f"data/c_fnirs-fmri-corr{_out_suffix}")

# ─── PARAMETERS ───────────────────────────────────────────────────────────────

N_PERM           = 1000
FDR_ALPHA        = 0.05
DIST_WARNING_MM  = 20.0
SHORT_CH_DIST_MM = 15.0
N_STANDARD_CH    = 42    # updated below after optional frontal masking
N_ROIS           = 122
DROP_FIRST_N     = 3
SPHERE_RADIUS_MM = 5.0

os.makedirs(OUT_DIR, exist_ok=True)

# ─── STEP 1: LOAD fNIRS, DROP FIRST 3 TIMEPOINTS ─────────────────────────────

print("=" * 60)
print("STEP 1: Loading fNIRS HbO data")
print("=" * 60)

# Short channels (0-indexed): [4,11,20,23,27,31,38,49] (MATLAB 1-indexed: [5,12,21,24,28,32,39,50])
SHORT_CH_IDX_0   = [4, 11, 20, 23, 27, 31, 38, 49]
standard_ch_mask = np.ones(50, dtype=bool)
standard_ch_mask[SHORT_CH_IDX_0] = False

fnirs_mat    = sio.loadmat(FNIRS_FILE)
zhbo_raw     = fnirs_mat['zhbo_TxRxS']
if zhbo_raw.shape[1] == 50:
    zhbo_TxRxS = zhbo_raw[:, standard_ch_mask, :]   # (T, 42, N_subj)
else:
    zhbo_TxRxS = zhbo_raw                            # already 42 channels
print(f"Loaded zhbo_TxRxS shape : {zhbo_TxRxS.shape}  (T x channels x subjects)")

zhbo_aligned = zhbo_TxRxS[DROP_FIRST_N:, :, :]  # (886, 42, N_subj)
print(f"After dropping {DROP_FIRST_N} timepoints: {zhbo_aligned.shape}")

# ─── STEP 2: GROUP-MEAN fNIRS ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Computing group-mean fNIRS HbO")
print("=" * 60)

fnirs_mean = np.nanmean(zhbo_aligned, axis=2)    # (886, 42)
if FRONTAL_ONLY:
    fnirs_mean = fnirs_mean[:, FRONTAL_CH_IDX_42]  # (886, 22)
print(f"Group-mean fNIRS shape: {fnirs_mean.shape}  (T x channels)")

# ─── STEP 3: LOAD fMRI GROUP-MEAN ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Loading fMRI group-mean BOLD")
print("=" * 60)

fmri_mean = np.load(FMRI_MEAN)                   # (886, 122)
print(f"Group-mean fMRI shape : {fmri_mean.shape}  (T x ROIs)")

if fnirs_mean.shape[0] != fmri_mean.shape[0]:
    raise ValueError(
        f"Timepoint mismatch: fNIRS={fnirs_mean.shape[0]}, fMRI={fmri_mean.shape[0]}"
    )

# ─── STEP 4: READ MNI COORDINATES ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Reading MNI coordinates from Standard_Channels.txt")
print("=" * 60)

# Comma-separated, no header
# Cols: 0=ch, 1=src, 2=det, 3-5=scalp_xyz, 6=dist_mm, 7-9=MNI_xyz
chan_df        = pd.read_csv(STDCHAN_FILE, header=None, sep=',')
standard_mask  = chan_df.iloc[:, 6] > SHORT_CH_DIST_MM
chan_standard  = chan_df[standard_mask].reset_index(drop=True)

print(f"Total rows            : {len(chan_df)}")
print(f"Standard channels     : {len(chan_standard)} (dist > {SHORT_CH_DIST_MM}mm, expected {N_STANDARD_CH})")

if len(chan_standard) != N_STANDARD_CH:
    print(f"*** WARNING: Expected {N_STANDARD_CH}, found {len(chan_standard)}. Check threshold. ***")

mni_coords = chan_standard.iloc[:, 7:10].values.astype(float)  # (42, 3)
if FRONTAL_ONLY:
    mni_coords = mni_coords[FRONTAL_CH_IDX_42]                 # (22, 3)
N_STANDARD_CH = len(mni_coords)
print(f"MNI coords shape      : {mni_coords.shape}")
print(f"First 3 channels:\n{mni_coords[:3].round(2)}")

# ─── STEP 5: ROI CENTROIDS ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Computing ROI centroids in MNI space")
print("=" * 60)

atlas_img     = nib.load(ATLAS_FILE)
atlas_data    = atlas_img.get_fdata().astype(int)
atlas_affine  = atlas_img.affine
atlas_shape   = atlas_img.shape

roi_labels    = np.unique(atlas_data)
roi_labels    = roi_labels[roi_labels > 0]
print(f"ROI labels found: {len(roi_labels)} (expected {N_ROIS})")

roi_centroids = np.zeros((len(roi_labels), 3))
for i, roi_id in enumerate(roi_labels):
    vox_idx           = np.argwhere(atlas_data == roi_id)
    vox_hom           = np.column_stack([vox_idx, np.ones(len(vox_idx))])
    mni_pts           = (atlas_affine @ vox_hom.T).T[:, :3]
    roi_centroids[i]  = mni_pts.mean(axis=0)

print(f"ROI centroids shape: {roi_centroids.shape}")

# ─── STEP 6: CHANNEL-TO-ROI MATCHING ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Matching channels to nearest ROI")
print("=" * 60)

diff             = mni_coords[:, np.newaxis, :] - roi_centroids[np.newaxis, :, :]
dist_matrix      = np.sqrt((diff ** 2).sum(axis=2))  # (42, 122)
nearest_roi_idx  = dist_matrix.argmin(axis=1)
nearest_roi_id   = roi_labels[nearest_roi_idx]
nearest_dist     = dist_matrix.min(axis=1)

print(f"\n{'Ch':>4}  {'MNI (x,y,z)':>24}  {'ROI':>5}  {'Dist(mm)':>9}  {'Flag'}")
print("-" * 62)
for ch in range(len(mni_coords)):
    flag = "*** >20mm" if nearest_dist[ch] > DIST_WARNING_MM else ""
    print(f"{ch+1:>4}  ({mni_coords[ch,0]:6.1f},{mni_coords[ch,1]:6.1f},"
          f"{mni_coords[ch,2]:6.1f})  {nearest_roi_id[ch]:>5}  "
          f"{nearest_dist[ch]:>9.2f}  {flag}")

# Collision check
unique_rois, counts = np.unique(nearest_roi_id, return_counts=True)
collisions = unique_rois[counts > 1]
if len(collisions) > 0:
    print(f"\n*** {len(collisions)} ROIs matched by multiple channels:")
    for roi in collisions:
        ch_list = np.where(nearest_roi_id == roi)[0] + 1
        print(f"  ROI {roi}: channels {ch_list.tolist()}")
else:
    print("\nNo collisions.")

match_df = pd.DataFrame({
    'channel':     np.arange(1, len(mni_coords) + 1),
    'mni_x':       mni_coords[:, 0].round(2),
    'mni_y':       mni_coords[:, 1].round(2),
    'mni_z':       mni_coords[:, 2].round(2),
    'matched_roi': nearest_roi_id,
    'distance_mm': nearest_dist.round(2),
    'flag':        ['dist>20mm' if d > DIST_WARNING_MM else '' for d in nearest_dist]
})
match_df.to_csv(os.path.join(OUT_DIR, "channel_roi_matches.csv"), index=False)

# ─── STEP 7: PEARSON r ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Computing Pearson r for each matched pair")
print("=" * 60)

grp_grp_corr_Rx1 = np.zeros(N_STANDARD_CH)

for ch in range(N_STANDARD_CH):
    roi_idx  = nearest_roi_idx[ch]
    hbo_ts   = fnirs_mean[:, ch]
    bold_ts  = fmri_mean[:, roi_idx]
    keep     = np.isfinite(hbo_ts) & np.isfinite(bold_ts)
    if keep.sum() < 3:
        grp_grp_corr_Rx1[ch] = np.nan
    else:
        grp_grp_corr_Rx1[ch], _ = stats.pearsonr(hbo_ts[keep], bold_ts[keep])

print(f"Mean r   : {np.nanmean(grp_grp_corr_Rx1):.4f}")
print(f"Median r : {np.nanmedian(grp_grp_corr_Rx1):.4f}")
print(f"Std r    : {np.nanstd(grp_grp_corr_Rx1):.4f}")
print(f"r values : {np.round(grp_grp_corr_Rx1, 3)}")

# ─── STEP 8: PHASE-RANDOMIZATION PERMUTATION TEST (nltools) ───────────────────

print("\n" + "=" * 60)
print(f"STEP 8: Phase-randomization permutation test ({N_PERM} iters, nltools)")
print("=" * 60)

null_grp_grp_corr_RxN  = np.zeros((N_STANDARD_CH, N_PERM))
grp_grp_corr_pval_Rx1  = np.zeros(N_STANDARD_CH)

for ch in range(N_STANDARD_CH):
    if (ch + 1) % 10 == 0:
        print(f"  Channel {ch+1}/{N_STANDARD_CH}...")

    roi_idx  = nearest_roi_idx[ch]
    hbo_ts   = fnirs_mean[:, ch]
    bold_ts  = fmri_mean[:, roi_idx]
    keep     = np.isfinite(hbo_ts) & np.isfinite(bold_ts)

    if keep.sum() < 3:
        null_grp_grp_corr_RxN[ch, :] = np.nan
        grp_grp_corr_pval_Rx1[ch]    = np.nan
        continue

    null_dist = []
    for i in range(N_PERM):
        bold_perm    = phase_randomize(bold_ts[keep], random_state=None)
        r_perm, _    = stats.pearsonr(hbo_ts[keep], bold_perm)
        null_dist.append(r_perm)

    null_grp_grp_corr_RxN[ch, :] = null_dist

    # one-tailed p-value (right tail)
    grp_grp_corr_pval_Rx1[ch] = (
        np.sum(np.array(null_dist) >= grp_grp_corr_Rx1[ch]) + 1
    ) / (N_PERM + 1)

# ─── STEP 9: FDR CORRECTION ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9: FDR correction (fdr_bh)")
print("=" * 60)

grp_grp_corr_sigmask_Rx1, grp_grp_corr_pval_corrected_Rx1, _, _ = multipletests(
    grp_grp_corr_pval_Rx1, FDR_ALPHA, 'fdr_bh'
)
grp_grp_corr_sigroi     = np.where(grp_grp_corr_pval_corrected_Rx1 < FDR_ALPHA)[0] + 1
grp_grp_corr_pmasked_Rx1 = grp_grp_corr_Rx1 * grp_grp_corr_sigmask_Rx1

n_sig = grp_grp_corr_sigmask_Rx1.sum()
print(f"Significant: {n_sig}/{N_STANDARD_CH} channels (FDR q<{FDR_ALPHA})")
print(f"Sig channel indices: {grp_grp_corr_sigroi}")
print(grp_grp_corr_sigmask_Rx1)

# ─── STEP 10: SAVE AS .mat ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 10: Saving results as .mat")
print("=" * 60)

out_mat = os.path.join(OUT_DIR, "fnirs_fmri_corr.mat")
sio.savemat(out_mat, {
    'grp_grp_corr_Rx1':              grp_grp_corr_Rx1,
    'null_grp_grp_corr_RxN':         null_grp_grp_corr_RxN,
    'grp_grp_corr_pval_Rx1':         grp_grp_corr_pval_Rx1,
    'grp_grp_corr_pval_corrected_Rx1': grp_grp_corr_pval_corrected_Rx1,
    'grp_grp_corr_sigmask_Rx1':      grp_grp_corr_sigmask_Rx1.astype(int),
    'grp_grp_corr_sigroi':           grp_grp_corr_sigroi,
    'grp_grp_corr_pmasked_Rx1':      grp_grp_corr_pmasked_Rx1,
    'mni_coords':                    mni_coords,
    'nearest_roi_id':                nearest_roi_id,
    'nearest_dist_mm':               nearest_dist,
})
print(f"Saved: {out_mat}")

# ─── STEP 11: BAR PLOT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 11: Bar plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#b22222' if s else '#7a7a7a' for s in grp_grp_corr_sigmask_Rx1]
ax.bar(np.arange(1, N_STANDARD_CH + 1), grp_grp_corr_Rx1, color=colors, width=0.7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel("fNIRS channel", fontsize=12)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title(
    f"Phase 3: fNIRS-fMRI correlation at matching locations\n"
    f"red=significant (FDR q<{FDR_ALPHA}), "
    f"n={n_sig}/{N_STANDARD_CH}, "
    f"median r={np.nanmedian(grp_grp_corr_Rx1):.3f}",
    fontsize=11
)
ax.set_xticks(np.arange(1, N_STANDARD_CH + 1))
ax.set_xticklabels(np.arange(1, N_STANDARD_CH + 1), fontsize=7)

# significance markers: ** for q<0.01, * for q<0.05
for ch in range(N_STANDARD_CH):
    if grp_grp_corr_pval_corrected_Rx1[ch] < 0.01:
        ax.text(ch+1, grp_grp_corr_Rx1[ch]+0.01, '**', ha='center', va='bottom', fontsize=8)
    elif grp_grp_corr_sigmask_Rx1[ch]:
        ax.text(ch+1, grp_grp_corr_Rx1[ch]+0.01, '*',  ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "fnirs_fmri_corr_barplot.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"Bar plot saved: {fig_path}")

# ─── STEP 11b: HEATMAP — matching ROI r + global grey matter control ──────────
# Replicates the heatmap from Gao et al. supplementary / old pipeline
# b_unimodal-bimodal_fc_matrix.ipynb
# Y-axis: fNIRS channels, X-axis: Matching ROI | Global grey matter
# Colormap: blue→white→red, -0.5 to 0.5, significance markers on right

print("\nGenerating correlation heatmap (matching + global control)...")

# Build the matrix: (N_CH x 2)
# col 0 = matching ROI r (already computed)
# col 1 = global grey matter r (need to compute now from fmri_mean)
global_ts_for_heatmap = fmri_mean.mean(axis=1)   # (886,)
global_r_heatmap = np.zeros(N_STANDARD_CH)
for ch in range(N_STANDARD_CH):
    hbo_ts = fnirs_mean[:, ch]
    keep   = np.isfinite(hbo_ts) & np.isfinite(global_ts_for_heatmap)
    if keep.sum() < 3:
        global_r_heatmap[ch] = np.nan
    else:
        global_r_heatmap[ch], _ = stats.pearsonr(hbo_ts[keep], global_ts_for_heatmap[keep])

# Stack into (N_CH x 2) matrix
corr_matrix = np.column_stack([grp_grp_corr_Rx1, global_r_heatmap])  # (42, 2)

# Custom diverging colormap: blue → white → red
from matplotlib.colors import LinearSegmentedColormap
cmap_bwr = LinearSegmentedColormap.from_list(
    'bwr_gao', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'], N=256)

fig_h, ax_h = plt.subplots(figsize=(3, 12))

im = ax_h.imshow(corr_matrix, cmap=cmap_bwr, vmin=-0.5, vmax=0.5,
                  aspect='auto', interpolation='none')

# x-axis labels
ax_h.set_xticks([0, 1])
ax_h.set_xticklabels(['Matching', 'Yeo Global'], fontsize=10, rotation=45,
                      ha='left', va='bottom')
ax_h.xaxis.set_ticks_position('top')
ax_h.xaxis.set_label_position('top')
ax_h.set_xlabel('fMRI ROI', fontsize=11)

# y-axis: channel numbers
ax_h.set_yticks(np.arange(N_STANDARD_CH))
ax_h.set_yticklabels(np.arange(1, N_STANDARD_CH + 1), fontsize=7)
ax_h.set_ylabel('fNIRS Channel', fontsize=11)

# significance markers on right side (matching ROI only, col 0)
for ch in range(N_STANDARD_CH):
    if grp_grp_corr_pval_corrected_Rx1[ch] < 0.01:
        ax_h.text(1.6, ch, '**', ha='left', va='center', fontsize=7, color='black')
    elif grp_grp_corr_sigmask_Rx1[ch]:
        ax_h.text(1.6, ch, '*',  ha='left', va='center', fontsize=7, color='black')

# grid lines between channels
ax_h.set_yticks(np.arange(-0.5, N_STANDARD_CH, 1), minor=True)
ax_h.yaxis.grid(True, which='minor', color='white', linewidth=0.5)
ax_h.set_xticks([-0.5, 0.5, 1.5], minor=True)
ax_h.xaxis.grid(True, which='minor', color='white', linewidth=1)

# colorbar
cbar = plt.colorbar(im, ax=ax_h, shrink=0.3, pad=0.15)
cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
heatmap_path = os.path.join(OUT_DIR, "fnirs_fmri_corr_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {heatmap_path}")

# ─── STEP 12: NIFTI SANITY CHECK FILES ────────────────────────────────────────
# Two volumes for FSLeyes spatial verification:
#   fnirs_channels_spheres.nii.gz  — 5mm sphere at each channel MNI coord
#   roi_matches_labeled.nii.gz     — matched ROIs labeled by channel number
#
# Usage (local):
#   scp midway3:.../data/c_fnirs-fmri-corr/*.nii.gz .
#   scp midway3:.../data/a_fmri-roi-ts/atlas_122_resampled.nii.gz .
#   fsleyes atlas_122_resampled.nii.gz roi_matches_labeled.nii.gz fnirs_channels_spheres.nii.gz
#
# Check: channel sphere (scalp) should sit above its same-colored matched ROI

print("\n" + "=" * 60)
print("STEP 12: Generating NIfTI sanity check files")
print("=" * 60)

affine_inv = np.linalg.inv(atlas_affine)
voxel_size = np.sqrt((atlas_affine[:3, :3] ** 2).sum(axis=0)).mean()
r_vox      = int(np.ceil(SPHERE_RADIUS_MM / voxel_size))

# File 1: channel spheres
sphere_vol = np.zeros(atlas_shape, dtype=np.int16)
for ch in range(len(mni_coords)):
    mni_hom   = np.append(mni_coords[ch], 1.0)
    vox_coord = (affine_inv @ mni_hom)[:3].round().astype(int)
    cx, cy, cz = vox_coord
    for dx in range(-r_vox, r_vox+1):
        for dy in range(-r_vox, r_vox+1):
            for dz in range(-r_vox, r_vox+1):
                if dx**2 + dy**2 + dz**2 <= r_vox**2:
                    nx, ny, nz = cx+dx, cy+dy, cz+dz
                    if (0 <= nx < atlas_shape[0] and
                        0 <= ny < atlas_shape[1] and
                        0 <= nz < atlas_shape[2]):
                        sphere_vol[nx, ny, nz] = ch + 1

nib.save(nib.Nifti1Image(sphere_vol, atlas_affine, atlas_img.header),
         os.path.join(OUT_DIR, "fnirs_channels_spheres.nii.gz"))
print(f"Channel spheres saved.")

# File 2: matched ROIs labeled by channel number
roi_labeled_vol = np.zeros(atlas_shape, dtype=np.int16)
for ch in range(len(mni_coords)):
    roi_labeled_vol[atlas_data == nearest_roi_id[ch]] = ch + 1

nib.save(nib.Nifti1Image(roi_labeled_vol, atlas_affine, atlas_img.header),
         os.path.join(OUT_DIR, "roi_matches_labeled.nii.gz"))
print(f"Matched ROIs NIfTI saved.")

# ─── SUPP TABLE S2: CONTROL ANALYSIS ─────────────────────────────────────────
# Correlate each fNIRS channel against:
#   1. Primary auditory cortex  (ROIs 10 + 67 — LH/RH SomMotB_Aud)
#   2. Primary visual cortex    (ROIs 1-5 + 58-62 — LH/RH visual)
#   3. Global grey matter signal (mean across all 122 ROIs)
# Per Gao et al.: none should be significant → confirms spatial specificity

print("\n" + "=" * 60)
print("SUPP TABLE S2: Control analysis")
print("=" * 60)

AUD_ROI_IDX = [9, 66]                        # ROIs 10, 67 (0-indexed)
VIS_ROI_IDX = [0,1,2,3,4, 57,58,59,60,61]   # ROIs 1-5, 58-62 (0-indexed)

aud_ts    = fmri_mean[:, AUD_ROI_IDX].mean(axis=1)
vis_ts    = fmri_mean[:, VIS_ROI_IDX].mean(axis=1)
global_ts = fmri_mean.mean(axis=1)

ctrl_names = ['primary_auditory', 'primary_visual', 'global_grey_matter']
ctrl_ts    = [aud_ts, vis_ts, global_ts]
ctrl_results = {}

for ctrl_name, ctrl_bold in zip(ctrl_names, ctrl_ts):
    print(f"\n  Control: {ctrl_name}")

    ctrl_r = np.zeros(N_STANDARD_CH)
    for ch in range(N_STANDARD_CH):
        hbo_ts = fnirs_mean[:, ch]
        keep   = np.isfinite(hbo_ts) & np.isfinite(ctrl_bold)
        if keep.sum() < 3:
            ctrl_r[ch] = np.nan
        else:
            ctrl_r[ch], _ = stats.pearsonr(hbo_ts[keep], ctrl_bold[keep])

    null_r = np.zeros((N_STANDARD_CH, N_PERM))
    pvals  = np.zeros(N_STANDARD_CH)
    for ch in range(N_STANDARD_CH):
        hbo_ts = fnirs_mean[:, ch]
        keep   = np.isfinite(hbo_ts) & np.isfinite(ctrl_bold)
        if keep.sum() < 3:
            null_r[ch, :] = np.nan
            pvals[ch]     = np.nan
            continue
        nd = []
        for i in range(N_PERM):
            bold_perm = phase_randomize(ctrl_bold[keep], random_state=None)
            r_p, _    = stats.pearsonr(hbo_ts[keep], bold_perm)
            nd.append(r_p)
        null_r[ch, :] = nd
        pvals[ch]     = (np.sum(np.array(nd) >= ctrl_r[ch]) + 1) / (N_PERM + 1)

    reject_ctrl, qvals_ctrl, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method='fdr_bh')
    n_sig_ctrl = reject_ctrl.sum()
    print(f"    Median r = {np.nanmedian(ctrl_r):.4f}")
    print(f"    Significant: {n_sig_ctrl}/{N_STANDARD_CH} (FDR q<{FDR_ALPHA})")

    ctrl_results[ctrl_name] = {
        'r': ctrl_r, 'p': pvals, 'q': qvals_ctrl,
        'significant': reject_ctrl.astype(int),
        'median_r': np.nanmedian(ctrl_r), 'n_sig': n_sig_ctrl,
    }

# Save Supp Table S2 CSV
supp_s2_rows = []
for ch in range(N_STANDARD_CH):
    row = {'channel': ch + 1}
    for ctrl_name in ctrl_names:
        row[f'{ctrl_name}_r']   = round(ctrl_results[ctrl_name]['r'][ch], 4)
        row[f'{ctrl_name}_q']   = round(ctrl_results[ctrl_name]['q'][ch], 4)
        row[f'{ctrl_name}_sig'] = ctrl_results[ctrl_name]['significant'][ch]
    supp_s2_rows.append(row)

supp_s2_df = pd.DataFrame(supp_s2_rows)
supp_s2_df.to_csv(os.path.join(OUT_DIR, "supp_table_s2_control_analysis.csv"), index=False)
print(f"\nSupp Table S2 saved.")

print("\nSummary:")
print(f"{'Control':<25} {'Median r':>10} {'N sig':>8}")
print("-" * 45)
for ctrl_name in ctrl_names:
    print(f"{ctrl_name:<25} {ctrl_results[ctrl_name]['median_r']:>10.4f} "
          f"{ctrl_results[ctrl_name]['n_sig']:>8}/{N_STANDARD_CH}")

# Append to main .mat
d_main = sio.loadmat(os.path.join(OUT_DIR, "fnirs_fmri_corr.mat"))
for ctrl_name in ctrl_names:
    d_main[f'ctrl_{ctrl_name}_r']   = ctrl_results[ctrl_name]['r']
    d_main[f'ctrl_{ctrl_name}_q']   = ctrl_results[ctrl_name]['q']
    d_main[f'ctrl_{ctrl_name}_sig'] = ctrl_results[ctrl_name]['significant']
sio.savemat(os.path.join(OUT_DIR, "fnirs_fmri_corr.mat"), d_main)
print("Control results appended to fnirs_fmri_corr.mat")

print("\nDone. Phase 3 complete.")
print(f"All outputs in: {OUT_DIR}")