"""
k_parse_timing_behavior.py
--------------------------
Extract block/trial onset timing and behavioral metrics from raw fNIRS
trigger (.tri) files and PsychoPy output (.csv) files.

Produces outputs used by downstream scripts:
  1. N-back block timing (at resampled 1 Hz) — used by m_task_glm_contrast.py
     to build GLM design matrices.
  2. CPT trial-level event timing (at resampled 1 Hz) — saved for reference;
     CPT has no block design, so the GLM in m_ uses only N-back.
  3. Behavioral summary (accuracy + d-prime per subject) — used by
     n_pred_fc_cpm.py as regression targets.

Session-to-task mapping:
  Folder suffix  Task
  _001           Nback
  _002           CPT1
  _003           CPTMEM1  (not processed here)
  _004           CPT2
  _005           CPTMEM2  (not processed here)
  _006           NNW      (not processed here)

Trigger codes in _fixed_lsl.tri:
  Nback  (session _001): 99 = task start, 1 = 1-back block onset, 2 = 2-back block onset
  CPT    (sessions _002, _004): 99 = task start (incl. practice), 1 = standard trial, 2 = target trial

Raw fNIRS sampling rate : ~5.0863 Hz
Resampled rate (b1 output): 1 Hz  (matches fMRI TR; 300 TRs per task session)

Inputs:
  - /project/ycleong/datasets/CogTasks-NNW_judy/  (raw fNIRS + .tri files)
  - /project/ycleong/datasets/CogTasks-NNW_judy/psychopy_data/nback/
  - /project/ycleong/datasets/CogTasks-NNW_judy/psychopy_data/cpt_dual/

Outputs (data/k_task-timing/):
  - nback_block_timing.json   : {subj_id → [{block_idx, condition, onset_s,
                                  offset_s, onset_TR, offset_TR}, ...]}
  - cpt1_trial_timing.json    : {subj_id → [{onset_s, onset_TR, trial_type}, ...]}
  - cpt2_trial_timing.json    : same for CPT2
  - behavioral_scores.csv     : columns: subj_id, nback_acc_1back, nback_acc_2back,
                                  nback_dprime_1back, nback_dprime_2back,
                                  nback_dprime_overall, cpt1_dprime, cpt1_acc,
                                  cpt2_dprime, cpt2_acc

Usage:
  python scripts/k_parse_timing_behavior.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

# ── paths ─────────────────────────────────────────────────────────────────────
BASE          = "/project/ycleong/users/judycchen/prediction-proj"
RAW_BASE      = "/project/ycleong/datasets/CogTasks-NNW_judy"
NBACK_DIR     = os.path.join(RAW_BASE, "psychopy_data/nback")
CPT_DIR       = os.path.join(RAW_BASE, "psychopy_data/cpt_dual")
OUT_DIR       = os.path.join(BASE, "data/k_task-timing")
os.makedirs(OUT_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────
RAW_HZ    = 5.0863
RESAMP_HZ = 1.0
N_TRS     = 300

# ── subject list ──────────────────────────────────────────────────────────────
subj_dirs = sorted(glob.glob(os.path.join(RAW_BASE, 'p*_fnirs_cogTasks')))
subj_ids  = [os.path.basename(d).split('_')[0] for d in subj_dirs]   # ['p01', 'p02', ...]
print(f"Found {len(subj_ids)} subjects: {subj_ids}")

# ── helpers ───────────────────────────────────────────────────────────────────

def find_tri(subj_id, session_suffix):
    """Return path to _fixed_lsl.tri for a session; fall back to _lsl.tri."""
    subj_dir  = os.path.join(RAW_BASE, f'{subj_id}_fnirs_cogTasks')
    sess_dirs = sorted(glob.glob(os.path.join(subj_dir, f'*_{session_suffix}')))
    if not sess_dirs:
        return None
    sess_dir = sess_dirs[0]
    fixed    = glob.glob(os.path.join(sess_dir, '*_fixed_lsl.tri'))
    if fixed:
        return fixed[0]
    fallback = glob.glob(os.path.join(sess_dir, '*_lsl.tri'))
    return fallback[0] if fallback else None

def parse_tri(tri_path):
    """Parse .tri file → DataFrame with columns [timestamp, raw_sample, trigger]."""
    rows = []
    with open(tri_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) < 3:
                continue
            try:
                rows.append({
                    'timestamp':  parts[0],
                    'raw_sample': int(parts[1]),
                    'trigger':    int(parts[2]),
                })
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df[df['trigger'] != 0].reset_index(drop=True)
    return df

def raw_to_resamp(raw_sample, task_start_raw):
    """Convert a raw sample index to resampled 1 Hz TR index and seconds."""
    offset_raw  = raw_sample - task_start_raw
    time_s      = offset_raw / RAW_HZ
    onset_tr    = int(round(time_s * RESAMP_HZ))
    return onset_tr, time_s

def compute_dprime(hits, misses, fas, crs):
    hit_rate = np.clip(hits / max(hits + misses, 1), 0.01, 0.99)
    fa_rate  = np.clip(fas  / max(fas  + crs,   1), 0.01, 0.99)
    return float(norm.ppf(hit_rate) - norm.ppf(fa_rate))

# ── SECTION 1: N-back block timing ───────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1: N-back block timing")
print("=" * 60)

nback_timing = {}

for subj_id in subj_ids:
    tri_path = find_tri(subj_id, '001')
    if tri_path is None:
        print(f"  {subj_id}: no Nback .tri found — skipping")
        continue

    events = parse_tri(tri_path)
    if events.empty or 99 not in events['trigger'].values:
        print(f"  {subj_id}: no task-start trigger (99) — skipping")
        continue

    task_start_raw = int(events[events['trigger'] == 99].iloc[0]['raw_sample'])
    block_events   = events[events['trigger'].isin([1, 2])].reset_index(drop=True)

    if block_events.empty:
        print(f"  {subj_id}: no block onset triggers (1/2) — skipping")
        continue

    blocks = []
    for i, row in block_events.iterrows():
        onset_tr, onset_s = raw_to_resamp(int(row['raw_sample']), task_start_raw)

        # offset = next block onset - 1 TR, or N_TRS for the last block
        later = block_events[block_events['raw_sample'] > row['raw_sample']]
        if len(later) > 0:
            offset_tr, _ = raw_to_resamp(int(later.iloc[0]['raw_sample']), task_start_raw)
            offset_tr -= 1
        else:
            offset_tr = N_TRS

        onset_tr  = max(0, onset_tr)
        offset_tr = min(N_TRS, offset_tr)

        blocks.append({
            'block_idx':  i,
            'condition':  '1back' if row['trigger'] == 1 else '2back',
            'onset_s':    round(onset_s, 3),
            'offset_s':   round(offset_tr / RESAMP_HZ, 3),
            'onset_TR':   onset_tr,
            'offset_TR':  offset_tr,
        })

    nback_timing[subj_id] = blocks
    n1 = sum(1 for b in blocks if b['condition'] == '1back')
    n2 = sum(1 for b in blocks if b['condition'] == '2back')
    print(f"  {subj_id}: {len(blocks)} blocks  (1-back={n1}, 2-back={n2})")

# ── SECTION 2: CPT trial-level timing ────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: CPT trial-level timing (standard=1, target=2)")
print("=" * 60)

cpt_timing = {'CPT1': {}, 'CPT2': {}}

for subj_id in subj_ids:
    for task_name, sess_suffix in [('CPT1', '002'), ('CPT2', '004')]:
        tri_path = find_tri(subj_id, sess_suffix)
        if tri_path is None:
            continue

        events = parse_tri(tri_path)
        if events.empty or 99 not in events['trigger'].values:
            continue

        task_start_raw = int(events[events['trigger'] == 99].iloc[0]['raw_sample'])
        trial_events   = events[events['trigger'].isin([1, 2])].reset_index(drop=True)

        trials = []
        for _, row in trial_events.iterrows():
            onset_tr, onset_s = raw_to_resamp(int(row['raw_sample']), task_start_raw)
            trials.append({
                'onset_s':    round(onset_s, 3),
                'onset_TR':   onset_tr,
                'trial_type': 'standard' if row['trigger'] == 1 else 'target',
            })

        cpt_timing[task_name][subj_id] = trials

    n1 = len(cpt_timing['CPT1'].get(subj_id, []))
    n2 = len(cpt_timing['CPT2'].get(subj_id, []))
    if n1 or n2:
        print(f"  {subj_id}: CPT1={n1} trials, CPT2={n2} trials")

# ── SECTION 3: N-back behavioral scores ──────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: N-back behavioral scores")
print("=" * 60)

nback_scores = {}

for subj_id in subj_ids:
    subj_num = subj_id.replace('p', '')   # 'p02' → '02'

    # prefer clean filename (e.g. 02_N-back_1.csv) over repeat run (02_N-back_1_1.csv)
    csv_files  = sorted(glob.glob(os.path.join(NBACK_DIR, f'{subj_num}_N-back_*.csv')))
    csv_clean  = [f for f in csv_files if not f.endswith('_1_1.csv')]
    csv_path   = csv_clean[0] if csv_clean else (csv_files[0] if csv_files else None)

    if csv_path is None:
        print(f"  {subj_id}: no Nback PsychoPy CSV — skipping")
        continue

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # keep only trial rows (corr_resp present)
    trial_rows = df[df['corr_resp'].notna()].copy()

    def get_condition(row):
        if pd.notna(row.get('Block1_1back.thisN')): return '1back'
        if pd.notna(row.get('Block2_1back.thisN')): return '1back'
        if pd.notna(row.get('Block1_2back.thisN')): return '2back'
        if pd.notna(row.get('Block2_2back.thisN')): return '2back'
        return None

    trial_rows = trial_rows.copy()
    trial_rows['condition'] = trial_rows.apply(get_condition, axis=1)
    trial_rows = trial_rows[trial_rows['condition'].notna()]

    scores = {'subj_id': subj_id}
    for cond in ['1back', '2back']:
        cond_rows   = trial_rows[trial_rows['condition'] == cond]
        if len(cond_rows) == 0:
            scores[f'nback_acc_{cond}']    = np.nan
            scores[f'nback_dprime_{cond}'] = np.nan
            continue

        scores[f'nback_acc_{cond}'] = float(cond_rows['key_resp_3.corr'].mean())

        target_rows    = cond_rows[cond_rows['corr_resp'] == 1]
        nontarget_rows = cond_rows[cond_rows['corr_resp'] == 0]
        hits   = int((target_rows['key_resp_3.corr']    == 1).sum())
        misses = int((target_rows['key_resp_3.corr']    == 0).sum())
        fas    = int((nontarget_rows['key_resp_3.corr'] == 0).sum())
        crs    = int((nontarget_rows['key_resp_3.corr'] == 1).sum())
        scores[f'nback_dprime_{cond}'] = compute_dprime(hits, misses, fas, crs)

    # overall d-prime across both conditions
    all_tgt    = trial_rows[trial_rows['corr_resp'] == 1]
    all_nontgt = trial_rows[trial_rows['corr_resp'] == 0]
    hits_all   = int((all_tgt['key_resp_3.corr']    == 1).sum())
    miss_all   = int((all_tgt['key_resp_3.corr']    == 0).sum())
    fas_all    = int((all_nontgt['key_resp_3.corr'] == 0).sum())
    crs_all    = int((all_nontgt['key_resp_3.corr'] == 1).sum())
    scores['nback_dprime_overall'] = compute_dprime(hits_all, miss_all, fas_all, crs_all)

    nback_scores[subj_id] = scores
    print(f"  {subj_id}: 1back acc={scores['nback_acc_1back']:.3f} d'={scores['nback_dprime_1back']:.3f} | "
          f"2back acc={scores['nback_acc_2back']:.3f} d'={scores['nback_dprime_2back']:.3f}")

# ── SECTION 4: CPT behavioral scores ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: CPT behavioral scores")
print("=" * 60)

cpt_scores = {}

for subj_id in subj_ids:
    subj_num = subj_id.replace('p', '')
    scores   = {'subj_id': subj_id}

    for sess_num, task_name in [('1', 'CPT1'), ('2', 'CPT2')]:
        csv_path = os.path.join(CPT_DIR, f'{subj_num}_AVA_CPT_{sess_num}.csv')
        if not os.path.exists(csv_path):
            scores[f'{task_name.lower()}_dprime'] = np.nan
            scores[f'{task_name.lower()}_acc']    = np.nan
            print(f"  {subj_id}: no {task_name} CSV")
            continue

        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # trial rows have hit/miss/FA/CR defined
        trial_rows = df[df[['hit', 'miss', 'FA', 'CR']].notna().any(axis=1)].copy()
        hits   = int(trial_rows['hit'].fillna(0).sum())
        misses = int(trial_rows['miss'].fillna(0).sum())
        fas    = int(trial_rows['FA'].fillna(0).sum())
        crs    = int(trial_rows['CR'].fillna(0).sum())

        scores[f'{task_name.lower()}_dprime'] = compute_dprime(hits, misses, fas, crs)
        total = hits + misses + fas + crs
        scores[f'{task_name.lower()}_acc']    = (hits + crs) / total if total > 0 else np.nan

    cpt_scores[subj_id] = scores
    d1 = scores.get('cpt1_dprime', np.nan)
    d2 = scores.get('cpt2_dprime', np.nan)
    print(f"  {subj_id}: CPT1 d'={d1:.3f}  CPT2 d'={d2:.3f}")

# ── SECTION 5: Save outputs ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: Saving outputs")
print("=" * 60)

# timing as JSON (nested dicts → easy to load in downstream Python scripts)
with open(os.path.join(OUT_DIR, 'nback_block_timing.json'), 'w') as f:
    json.dump(nback_timing, f, indent=2)
print(f"Saved nback_block_timing.json  ({len(nback_timing)} subjects)")

with open(os.path.join(OUT_DIR, 'cpt1_trial_timing.json'), 'w') as f:
    json.dump(cpt_timing['CPT1'], f, indent=2)
with open(os.path.join(OUT_DIR, 'cpt2_trial_timing.json'), 'w') as f:
    json.dump(cpt_timing['CPT2'], f, indent=2)
print(f"Saved cpt1/cpt2_trial_timing.json")

# behavioral scores CSV
all_rows = []
for subj_id in subj_ids:
    row = {}
    row.update(nback_scores.get(subj_id, {'subj_id': subj_id}))
    cpt = cpt_scores.get(subj_id, {})
    for k, v in cpt.items():
        if k != 'subj_id':
            row[k] = v
    row['subj_id'] = subj_id
    all_rows.append(row)

behav_df = pd.DataFrame(all_rows)
col_order = ['subj_id',
             'nback_acc_1back', 'nback_acc_2back',
             'nback_dprime_1back', 'nback_dprime_2back', 'nback_dprime_overall',
             'cpt1_dprime', 'cpt1_acc', 'cpt2_dprime', 'cpt2_acc']
col_order = [c for c in col_order if c in behav_df.columns]
behav_df  = behav_df[col_order]
behav_df.to_csv(os.path.join(OUT_DIR, 'behavioral_scores.csv'), index=False)
print(f"Saved behavioral_scores.csv  ({len(behav_df)} subjects)")

print("\nSummary:")
print(behav_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f"\nAll outputs in: {OUT_DIR}")
print("Done: k_parse_timing_behavior.py")
