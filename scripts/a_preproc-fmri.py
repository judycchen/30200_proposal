"""
a_preproc-fmri.py
-----------------
Phase 2: fMRI ROI time series extraction for NNW dataset (Anna's errts files)

Pipeline position: first script in prediction-proj
Input:  Anna's preprocessed errts files (already motion-corrected, nuisance-regressed,
        no GSR — no further filtering needed)
Output: data/a_fmri-roi-ts/

Decisions documented:
  - Atlas resampled TO BOLD space (not the reverse): atlas is a spatial index,
    BOLD is the signal. Resampling BOLD would fabricate sub-voxel data and
    compound interpolation artifacts. Atlas uses nearest-neighbor to preserve
    integer ROI labels.
  - No resampling of BOLD data to 3mm: ROI averaging makes voxel resolution
    irrelevant for the final output; we keep Anna's native resolution.
  - No additional filtering: Anna's errts files already have motion params (24),
    WM, and CSF regressed out via AFNI 3dDeconvolve (noGSR pipeline).
  - Z-scoring per ROI per subject: applied after extraction, before saving,
    consistent with downstream aPCR model expectations.
  - Crop to first 886 volumes: 889s movie - 3s removed upstream by Anna's
    pipeline = 886 volumes of actual movie content. Remaining ~60 volumes
    are post-movie waiting time.
  - Expected total T = 949: 889s movie + 60s wait, with first 3s removed
    upstream (so 886 + 60 + 3 = 949). Script prints actual T per subject
    so you can verify — crop proceeds regardless.

Outputs (all in data/a_fmri-roi-ts/):
  - atlas_122_resampled.nii.gz     : atlas in BOLD native space (saved once)
  - bold_TxRxS.npy                 : (886 x 122 x N_subj) all subjects
  - bold_groupmean_TxR.npy         : (886 x 122) group mean across subjects
  - subject_list.txt               : ordered list of included subjects
  - flagged_subjects.txt           : subjects with unexpected T or other issues

Usage:
  python a_preproc-fmri.py
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import scipy.io as sio
from nilearn.image import resample_to_img

HELPERS = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers"
sys.path.insert(0, HELPERS)
from isc import loo_isc, average_isc

# ─── PATHS ────────────────────────────────────────────────────────────────────

ERRTS_DIR  = "/project/ycleong/datasets/NNW_anna/preprocessed-errts"
ATLAS_PATH = ("/project/ycleong/users/judycchen/fnirs_fmri_models/data/sherlock"
              "/fmri/masks/Yeo_17N_114_Brainnetome_subcortical_8/2mm"
              "/all_122[nooverlap].nii.gz")
OUT_DIR    = "/project/ycleong/users/judycchen/prediction-proj/data/a_fmri-roi-ts"

# ─── PARAMETERS ───────────────────────────────────────────────────────────────

N_CROP     = 886   # volumes to keep: 889s movie - 3s removed upstream
N_ROIS     = 122
EXPECTED_T = 948   # this is 1s away from the expected 949 but it's alright

# ─── SETUP ────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ─── STEP 1: DISCOVER FILES ───────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Discovering errts files")
print("=" * 60)

files = sorted(glob.glob(os.path.join(ERRTS_DIR, "errts.*.tproject.nii*")))

if len(files) == 0:
    raise FileNotFoundError(
        f"No errts files found in {ERRTS_DIR}\n"
        "Expected pattern: errts.*.tproject.nii or errts.*.tproject.nii.gz"
    )

print(f"Found {len(files)} errts files:\n")
for f in files:
    print(f"  {os.path.basename(f)}")

# ─── STEP 2: INSPECT ONE FILE ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Inspecting first file to get BOLD space")
print("=" * 60)

bold_ref    = nib.load(files[0])
bold_shape  = bold_ref.shape        # (x, y, z, T)
bold_affine = bold_ref.affine
voxel_sizes = np.sqrt((bold_affine[:3, :3] ** 2).sum(axis=0))

print(f"File       : {os.path.basename(files[0])}")
print(f"Shape      : {bold_shape}  (x, y, z, timepoints)")
print(f"Voxel size : {voxel_sizes.round(3)} mm")
print(f"Affine     :\n{bold_affine.round(3)}")
print(f"\nExpected total T : {EXPECTED_T}")
print(f"Actual total T   : {bold_shape[3]}")

if bold_shape[3] != EXPECTED_T:
    print(f"\n*** WARNING: T={bold_shape[3]} differs from expected {EXPECTED_T} "
          f"by {bold_shape[3] - EXPECTED_T} volumes. ***")
    print("   Proceeding with crop to first 886 volumes regardless.")
else:
    print(f"\nTotal T matches expectation ({EXPECTED_T}). ")

# ─── STEP 3: RESAMPLE ATLAS TO BOLD SPACE ─────────────────────────────────────
# Atlas is the source; BOLD is the target reference.
# Nearest-neighbor preserves integer ROI labels — never use linear here.

print("\n" + "=" * 60)
print("STEP 3: Resampling atlas to BOLD space (nearest-neighbor)")
print("=" * 60)

atlas_img = nib.load(ATLAS_PATH)
atlas_vox = np.sqrt((atlas_img.affine[:3, :3] ** 2).sum(axis=0))
print(f"Atlas original shape  : {atlas_img.shape}")
print(f"Atlas original voxels : {atlas_vox.round(3)} mm")

# Extract 3D reference volume from BOLD (single timepoint)
bold_ref_3d = nib.Nifti1Image(
    bold_ref.get_fdata()[:, :, :, 0],
    bold_affine
)

# Resample atlas → BOLD space
atlas_resampled = resample_to_img(
    source_img=atlas_img,
    target_img=bold_ref_3d,
    interpolation='nearest',
    force_resample=True,
    copy_header=True
)

atlas_out_path = os.path.join(OUT_DIR, "atlas_122_resampled.nii.gz")
nib.save(atlas_resampled, atlas_out_path)

atlas_data = atlas_resampled.get_fdata().astype(int)
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels > 0]   # exclude background label 0

print(f"Atlas resampled shape : {atlas_resampled.shape}")
print(f"Atlas resampled voxels: {voxel_sizes.round(3)} mm  (matches BOLD)")
print(f"ROI labels found      : {len(roi_labels)} (expected {N_ROIS})")
if len(roi_labels) != N_ROIS:
    print(f"*** WARNING: Expected {N_ROIS} ROIs but found {len(roi_labels)}. "
          f"Check atlas filename — note the [nooverlap] bracket. ***")
print(f"Saved to: {atlas_out_path}")

# ─── STEP 4: EXTRACT ROI TIME SERIES PER SUBJECT ─────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Extracting ROI time series per subject")
print("=" * 60)

all_roi_ts  = []   # will become list of (886, 122) arrays
subject_ids = []
flagged     = []

for fpath in files:
    fname = os.path.basename(fpath)

    img  = nib.load(fpath)
    data = img.get_fdata()    # (x, y, z, T)
    T    = data.shape[3]

    print(f"\n  {fname}")
    print(f"    Original T = {T} volumes", end="")

    # Flag unexpected T but don't skip unless too short to crop
    flag_msg = None
    if T < N_CROP:
        flag_msg = f"T={T} < N_CROP={N_CROP}: cannot crop, SKIPPING"
        print(f"\n    *** {flag_msg} ***")
        flagged.append(f"{fname}: {flag_msg}")
        continue
    elif T != EXPECTED_T:
        flag_msg = f"T={T} differs from expected {EXPECTED_T} by {T - EXPECTED_T} vol"
        print(f"  *** {flag_msg} ***", end="")

    # Crop to first 886 volumes (movie content only)
    data_cropped = data[:, :, :, :N_CROP]   # (x, y, z, 886)
    print(f"  →  cropped to {N_CROP}")

    if flag_msg:
        flagged.append(f"{fname}: {flag_msg}")

    # Extract mean signal per ROI → (886, 122)
    roi_ts = np.full((N_CROP, len(roi_labels)), np.nan)

    for i, roi_id in enumerate(roi_labels):
        mask     = atlas_data == roi_id     # boolean (x, y, z)
        n_voxels = mask.sum()
        if n_voxels == 0:
            print(f"    *** WARNING: ROI {roi_id} has 0 voxels after resampling ***")
        else:
            roi_ts[:, i] = data_cropped[mask, :].mean(axis=0)

    # Z-score each ROI time series (mean=0, std=1 across time)
    roi_mean        = np.nanmean(roi_ts, axis=0, keepdims=True)
    roi_std         = np.nanstd(roi_ts, axis=0, keepdims=True)
    roi_std[roi_std == 0] = 1.0   # avoid divide-by-zero for flat signals
    roi_ts          = (roi_ts - roi_mean) / roi_std

    all_roi_ts.append(roi_ts)
    subject_ids.append(fname)
    print(f"    ROI matrix shape: {roi_ts.shape}  (timepoints × ROIs)  [z-scored]")

# ─── STEP 5: STACK AND SAVE ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Stacking subjects and saving outputs")
print("=" * 60)

# Stack to (886, 122, N_subj) — TxRxS convention matching downstream scripts
bold_TxRxS     = np.stack(all_roi_ts, axis=2)
bold_groupmean = np.nanmean(bold_TxRxS, axis=2)   # (886, 122)

print(f"bold_TxRxS shape   : {bold_TxRxS.shape}  (timepoints × ROIs × subjects)")
print(f"Group mean shape   : {bold_groupmean.shape}")
print(f"Subjects included  : {len(subject_ids)}")
print(f"Subjects flagged   : {len(flagged)}")

np.save(os.path.join(OUT_DIR, "bold_TxRxS.npy"),        bold_TxRxS)
np.save(os.path.join(OUT_DIR, "bold_groupmean_TxR.npy"), bold_groupmean)
sio.savemat(os.path.join(OUT_DIR, "bold_TxRxS.mat"), {"bold_TxRxS": bold_TxRxS})
sio.savemat(os.path.join(OUT_DIR, "bold_groupmean_TxR.mat"), {"bold_groupmean_TxR": bold_groupmean})

with open(os.path.join(OUT_DIR, "subject_list.txt"), "w") as f:
    for s in subject_ids:
        f.write(s + "\n")

with open(os.path.join(OUT_DIR, "flagged_subjects.txt"), "w") as f:
    if flagged:
        for s in flagged:
            f.write(s + "\n")
    else:
        f.write("No flagged subjects.\n")

print(f"\nOutputs saved to: {OUT_DIR}")
print("  bold_TxRxS.npy")
print("  bold_groupmean_TxR.npy")
print("  atlas_122_resampled.nii.gz")
print("  subject_list.txt")
print("  flagged_subjects.txt")

# ─── STEP 6: fMRI ISC ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Computing fMRI ISC (leave-one-out)")
print("=" * 60)

# loo_isc expects (S, T, R); bold_TxRxS is (T, R, S)
bold_SxTxR = bold_TxRxS.transpose(2, 0, 1)   # (N_subj, 886, 122)

# impute any NaN with 0 (data is z-scored so 0 = channel mean)
bold_SxTxR_imp = np.where(np.isnan(bold_SxTxR), 0.0, bold_SxTxR)

isc_SxR = loo_isc(bold_SxTxR_imp)             # (N_subj, 122)
isc_Sx1 = average_isc(isc_SxR, ax=1)          # (N_subj,)  mean ISC per subject
isc_Rx1 = average_isc(isc_SxR, ax=0)          # (122,)     mean ISC per ROI

print(f"\n{'Index':>6}  {'Subject':>45}  {'Mean ISC':>9}")
print("-" * 66)
for s, (sid, isc_val) in enumerate(zip(subject_ids, isc_Sx1)):
    print(f"{s+1:>6}  {sid:>45}  {isc_val:>9.4f}")
print(f"\nGrand mean ISC : {np.nanmean(isc_Sx1):.4f}")
print(f"Grand median ISC: {np.nanmedian(isc_Sx1):.4f}")

sio.savemat(os.path.join(OUT_DIR, "bold_isc.mat"), {
    "isc_SxR":     isc_SxR,
    "isc_Sx1":     isc_Sx1,
    "isc_Rx1":     isc_Rx1,
    "subject_ids": np.array(subject_ids),
})
print(f"\nISC saved: {os.path.join(OUT_DIR, 'bold_isc.mat')}")

print("\nDone.")
