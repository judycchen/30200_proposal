"""
l_apply_model_tasks.py
----------------------
Apply the final aPCR model (from j_train_final_model.py) to task fNIRS data
to generate predicted fMRI time series for each cognitive task session.

The forward pass is:
  fNIRS (T × 42) → center → fNIRS PCA → OLS → fMRI PCA inverse → predicted fMRI (T × 122)

Applied to five task sessions:
  Nback   (300 TRs, ~20 subjects)
  CPT1    (300 TRs, ~19 subjects)
  CPT2    (300 TRs, ~21 subjects)
  CPTMEM1 (300 TRs, ~20 subjects)
  CPTMEM2 (300 TRs, ~20 subjects)

Note: task fNIRS files in 5_excluded/ are already resampled to 1 Hz (300 TRs)
and already have short channels removed (42 channels). No further preprocessing
is applied here beyond NaN imputation.

Inputs:
  - data/j_final-model/apcr_final_model.npz
  - data/b_fnirs-preproc/5_excluded/all_{Task}_zhbo_TxRxS.mat

Outputs (data/l_task-pred-fmri/):
  - {Task}_pred_SxRxT.mat  : predicted fMRI, shape (n_subj, 122, T)
  - {Task}_pred_RxT.mat    : group-mean predicted fMRI, shape (122, T)
  - prediction_summary.csv : per-task n_subjects and signal quality check

Usage:
  python scripts/l_apply_model_tasks.py
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = "/project/ycleong/users/judycchen/prediction-proj"
MODEL_FILE  = os.path.join(BASE, "data/j_final-model/apcr_final_model.npz")
FNIRS_DIR   = os.path.join(BASE, "data/b_fnirs-preproc/5_excluded")
OUT_DIR     = os.path.join(BASE, "data/l_task-pred-fmri")
os.makedirs(OUT_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────
TASKS        = ['Nback', 'CPT1', 'CPT2', 'CPTMEM1', 'CPTMEM2']
SHORT_CH_IDX = [4, 11, 20, 23, 27, 31, 38, 49]   # 0-indexed; only applied if 50-ch input
N_ROIS       = 122

# ── helpers ───────────────────────────────────────────────────────────────────

def fill_nan_colmean(X):
    """Replace NaNs with column mean (0.0 if entire column is NaN)."""
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X = X.copy()
    X[nan_rows, nan_cols] = col_means[nan_cols]
    return X

def apply_apcr(fnirs_TxC, model):
    """
    Forward pass: fNIRS (T × C) → predicted fMRI (T × 122).
    model is a dict loaded from apcr_final_model.npz.
    """
    X_centered = fnirs_TxC - model['fnirs_mean']
    X_pc       = X_centered @ model['fnirs_components'].T        # (T, n_fnirs_pcs)
    y_pc_pred  = X_pc @ model['lr_coef'].T + model['lr_intercept']  # (T, n_fmri_pcs)
    y_pred     = y_pc_pred @ model['fmri_components'] + model['fmri_mean']  # (T, 122)
    return y_pred

# ── load model ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading model")
print("=" * 60)

model = dict(np.load(MODEL_FILE))
n_fnirs_pcs = int(model['fnirs_n_pcs'])
n_fmri_pcs  = int(model['fmri_n_pcs'])
print(f"Model loaded | fNIRS PCs: {n_fnirs_pcs}, fMRI PCs: {n_fmri_pcs}")

# ── apply model to each task ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Applying model to task fNIRS data")
print("=" * 60)

summary_rows = []

for task in TASKS:
    fnirs_path = os.path.join(FNIRS_DIR, f'all_{task}_zhbo_TxRxS.mat')
    if not os.path.exists(fnirs_path):
        print(f"\n{task}: file not found — skipping ({fnirs_path})")
        continue

    print(f"\n{'─'*40}")
    print(f"{task}")
    print(f"{'─'*40}")

    mat  = sio.loadmat(fnirs_path)
    zhbo = mat['zhbo_TxRxS']   # (T, 50_or_42, nS)

    # extract subject IDs stored by b2_exclude_fnirs.m
    if 'subj_ids_included' in mat:
        raw_ids  = mat['subj_ids_included'].flatten()
        subj_ids = [str(x.flat[0]) if hasattr(x, 'flat') else str(x) for x in raw_ids]
    else:
        subj_ids = [f's{i+1:02d}' for i in range(zhbo.shape[2])]
        print(f"  WARNING: subj_ids_included not in mat — using placeholder IDs")
    print(f"  Subject IDs: {subj_ids}")

    # drop short channels only if input has 50 channels (task files typically already 42)
    if zhbo.shape[1] == 50:
        standard_ch_mask = np.ones(50, dtype=bool)
        standard_ch_mask[SHORT_CH_IDX] = False
        zhbo = zhbo[:, standard_ch_mask, :]
        print(f"  Dropped short channels: 50 → 42")

    nT, nCh, nS = zhbo.shape
    print(f"  Shape: {zhbo.shape}  (T={nT}, channels={nCh}, subjects={nS})")

    pred_SxRxT = np.zeros((nS, N_ROIS, nT))

    for s in range(nS):
        fnirs_sub       = zhbo[:, :, s]          # (nT, 42)
        fnirs_sub       = fill_nan_colmean(fnirs_sub)
        pred_TxR        = apply_apcr(fnirs_sub, model)   # (nT, 122)
        pred_SxRxT[s]   = pred_TxR.T             # store as (122, nT)

    pred_mean_RxT = pred_SxRxT.mean(axis=0)      # (122, nT)
    mean_abs_val  = float(np.nanmean(np.abs(pred_SxRxT)))
    print(f"  pred_SxRxT: {pred_SxRxT.shape}  |  mean |pred|: {mean_abs_val:.4f}")

    # save per-subject and group-mean — include subj_ids for downstream alignment
    import json as _json
    sio.savemat(
        os.path.join(OUT_DIR, f'{task}_pred_SxRxT.mat'),
        {'pred_SxRxT': pred_SxRxT, 'n_subj': nS, 'n_rois': N_ROIS, 'n_trs': nT}
    )
    with open(os.path.join(OUT_DIR, f'{task}_subj_ids.json'), 'w') as _f:
        _json.dump(subj_ids, _f)
    sio.savemat(
        os.path.join(OUT_DIR, f'{task}_pred_RxT.mat'),
        {'pred_mean_RxT': pred_mean_RxT}
    )
    print(f"  Saved {task}_pred_SxRxT.mat and {task}_pred_RxT.mat")

    summary_rows.append({
        'task':           task,
        'n_subjects':     nS,
        'n_trs':          nT,
        'n_channels_in':  nCh,
        'mean_abs_pred':  round(mean_abs_val, 5),
    })

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'prediction_summary.csv'), index=False)
print(summary_df.to_string(index=False))
print(f"\nAll outputs in: {OUT_DIR}")
print("Done: l_apply_model_tasks.py")
