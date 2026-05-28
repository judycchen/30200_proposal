"""
compute_behavioral_metrics.py
------------------------------
Computes behavioral metrics for all 34 participants (p01-p34) across:
  - N-back: accuracy and RT (iSD, mean) per condition (1back, 2back)
  - CPT:    hit rate, commission error rate, d-prime, RT (iSD, mean) per session (1, 2)
  - CPT-MEM: recognition d-prime, hit/FA rates for faces and backgrounds per session (1, 2)

CPT task design (go/no-go):
  role_vis = 'freq_stim'   → GO trial  (press expected, ~90%)
  role_vis = 'infreq_stim' → NO-GO trial (withhold expected, ~10%)
  Hits = correct go presses; Commission errors = pressing on no-go trials

CPT-MEM old/new coding:
  Old stimuli = those whose filename appeared in the corresponding CPT session
  New stimuli = those whose filename did NOT appear in CPT
  Confidence >= 3 → "said old"; d-prime computed separately for faces and backgrounds

Outputs:
  behavioral_results/
    nback_metrics.csv
    cpt_metrics.csv
    cptmem_metrics.csv
    all_metrics_combined.csv
"""

import os
import ast
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_BASE = "/project/ycleong/datasets/CogTasks-NNW_judy/psychopy_data"
OUT_DIR   = os.path.join(DATA_BASE, "behavioral_results")
os.makedirs(OUT_DIR, exist_ok=True)

ALL_PIDS  = [f"{i:02d}" for i in range(1, 35)]

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_key(x):
    """Extract first key from PsychoPy list-string, e.g. \"['space']\" → 'space'."""
    if pd.isna(x) or str(x).strip() in ('nan', 'None', ''):
        return None
    try:
        v = ast.literal_eval(str(x))
        return v[0] if isinstance(v, list) else str(v)
    except Exception:
        return str(x).strip("[]'\" ")


def parse_rt(x):
    """Extract first RT value from PsychoPy list-string."""
    if pd.isna(x) or str(x).strip() in ('nan', 'None', ''):
        return None
    try:
        v = ast.literal_eval(str(x))
        return float(v[0]) if isinstance(v, list) else float(v)
    except Exception:
        return None


def dprime_sdt(hit_rate, fa_rate, n_signal, n_noise):
    """
    d-prime and criterion c with 0.5-trial correction for boundary cases.
    Returns (dprime, criterion) or (nan, nan) if inputs are invalid.
    """
    if any(np.isnan(x) for x in [hit_rate, fa_rate]):
        return np.nan, np.nan
    H = (hit_rate * n_signal + 0.5) / (n_signal + 1)
    F = (fa_rate  * n_noise  + 0.5) / (n_noise  + 1)
    dp   = norm.ppf(H) - norm.ppf(F)
    crit = -0.5 * (norm.ppf(H) + norm.ppf(F))
    return round(dp, 4), round(crit, 4)


def stim_basename(path_str):
    """Extract filename from a Windows or Unix path string."""
    if pd.isna(path_str):
        return None
    s = str(path_str).replace("\\", "/")
    return os.path.basename(s)


# ── N-back ────────────────────────────────────────────────────────────────────

def compute_nback(pid):
    path = os.path.join(DATA_BASE, "nback", f"{pid}_N-back_1.csv")
    if not os.path.exists(path):
        return {"pid": pid, "nback_file_missing": True}

    df = pd.read_csv(path)
    df = df[df['corr_resp'].notna()].copy()

    df['resp'] = df['key_resp_3.keys'].apply(parse_key)
    df['corr'] = df['corr_resp'].apply(parse_key)
    df['acc']  = (df['resp'] == df['corr']).astype(float)
    df['rt']   = df['key_resp_3.rt'].apply(parse_rt)

    def get_cond(row):
        if pd.notna(row.get('Block1_1back.thisN')) or pd.notna(row.get('Block2_1back.thisN')):
            return '1back'
        if pd.notna(row.get('Block1_2back.thisN')) or pd.notna(row.get('Block2_2back.thisN')):
            return '2back'
        return None

    df['cond'] = df.apply(get_cond, axis=1)

    out = {"pid": pid}
    for cond in ['1back', '2back']:
        sub     = df[df['cond'] == cond]
        correct = sub[sub['acc'] == 1]
        rts     = correct['rt'].dropna()

        out[f'nback_{cond}_n_trials']  = len(sub)
        out[f'nback_{cond}_accuracy']  = round(sub['acc'].mean(), 4) if len(sub) > 0 else np.nan
        out[f'nback_{cond}_rt_mean']   = round(rts.mean(),         4) if len(rts) > 1 else np.nan
        out[f'nback_{cond}_rt_isd']    = round(rts.std(ddof=1),    4) if len(rts) > 1 else np.nan
        out[f'nback_{cond}_n_correct'] = len(correct)

    return out


# ── CPT ───────────────────────────────────────────────────────────────────────

def compute_cpt(pid, session):
    path = os.path.join(DATA_BASE, "cpt_dual", f"{pid}_AVA_CPT_{session}.csv")
    if not os.path.exists(path):
        return {f"cpt{session}_file_missing": True}

    df = pd.read_csv(path)
    df = df[df['role_vis'].notna()].copy()

    df['pressed'] = df['key_resp.keys'].notna() & (df['key_resp.keys'].astype(str) != 'nan')
    df['rt']      = df['key_resp.rt'].apply(parse_rt)

    go   = df[df['role_vis'] == 'freq_stim']    # press expected
    nogo = df[df['role_vis'] == 'infreq_stim']  # withhold expected

    if len(go) == 0 or len(nogo) == 0:
        return {f"cpt{session}_no_trials": True}

    hit_rate  = go['pressed'].mean()
    comm_rate = nogo['pressed'].mean()
    dp, crit  = dprime_sdt(hit_rate, comm_rate, len(go), len(nogo))

    # RT only for correct go responses
    go_correct_rt = go[go['pressed']]['rt'].dropna()

    return {
        f'cpt{session}_n_go':             len(go),
        f'cpt{session}_n_nogo':           len(nogo),
        f'cpt{session}_hit_rate':         round(hit_rate,  4),
        f'cpt{session}_omission_rate':    round(1 - hit_rate, 4),
        f'cpt{session}_commission_rate':  round(comm_rate, 4),
        f'cpt{session}_cr_rate':          round(1 - comm_rate, 4),
        f'cpt{session}_dprime':           dp,
        f'cpt{session}_criterion':        crit,
        f'cpt{session}_rt_mean':          round(go_correct_rt.mean(),      4) if len(go_correct_rt) > 1 else np.nan,
        f'cpt{session}_rt_isd':           round(go_correct_rt.std(ddof=1), 4) if len(go_correct_rt) > 1 else np.nan,
        f'cpt{session}_n_go_correct':     int(go['pressed'].sum()),
    }


# ── CPT-MEM ───────────────────────────────────────────────────────────────────

def get_cpt_stim_set(pid, session):
    """Return (face_filenames, background_filenames) shown during CPT session."""
    path = os.path.join(DATA_BASE, "cpt_dual", f"{pid}_AVA_CPT_{session}.csv")
    if not os.path.exists(path):
        return set(), set()
    df = pd.read_csv(path)
    df = df[df['role_vis'].notna()].copy()
    faces = set(df['presented_stim_vis'].apply(stim_basename).dropna())
    bgs   = set(df['back_image'].apply(stim_basename).dropna())
    return faces, bgs


def compute_cptmem(pid, session):
    path = os.path.join(DATA_BASE, "cpt_mem", f"{pid}_AV_CPT_mem_{session}.csv")
    if not os.path.exists(path):
        return {f"cptmem{session}_file_missing": True}

    df = pd.read_csv(path)
    if 'presented_stim' not in df.columns:
        return {f"cptmem{session}_file_corrupt": True}
    df = df[df['presented_stim'].notna()].copy()
    if len(df) == 0:
        return {f"cptmem{session}_no_trials": True}

    cpt_faces, cpt_bgs = get_cpt_stim_set(pid, session)

    df['stim_name'] = df['presented_stim'].apply(stim_basename)
    df['is_face']   = df['presented_stim'].str.contains('face', case=False, na=False)
    df['is_bg']     = df['presented_stim'].str.contains('Indoor|Outdoor|background', case=False, na=False)
    df['is_old']    = df.apply(
        lambda r: (r['stim_name'] in cpt_faces) or (r['stim_name'] in cpt_bgs), axis=1
    )

    df['conf']     = pd.to_numeric(df['key_resp.keys'].apply(parse_key), errors='coerce')
    df['said_old'] = (df['conf'] >= 3)

    out = {}
    for stim_type, mask in [('face', df['is_face']), ('bg', df['is_bg'])]:
        sub = df[mask]
        old = sub[sub['is_old']]
        new = sub[~sub['is_old']]

        if len(old) > 0 and len(new) > 0:
            H = old['said_old'].mean()
            F = new['said_old'].mean()
            dp, crit = dprime_sdt(H, F, len(old), len(new))
        else:
            H = F = dp = crit = np.nan

        out[f'cptmem{session}_{stim_type}_n_old']     = len(old)
        out[f'cptmem{session}_{stim_type}_n_new']     = len(new)
        out[f'cptmem{session}_{stim_type}_hit_rate']  = round(H,    4) if not np.isnan(H) else np.nan
        out[f'cptmem{session}_{stim_type}_fa_rate']   = round(F,    4) if not np.isnan(F) else np.nan
        out[f'cptmem{session}_{stim_type}_dprime']    = dp
        out[f'cptmem{session}_{stim_type}_criterion'] = crit
        out[f'cptmem{session}_{stim_type}_mean_conf'] = round(sub['conf'].mean(), 4) if len(sub) > 0 else np.nan

    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("Computing behavioral metrics for all participants...")
print(f"Output directory: {OUT_DIR}\n")

nback_rows  = []
cpt_rows    = []
cptmem_rows = []

for pid in ALL_PIDS:
    print(f"  Processing p{pid}...", end=" ")

    nb = compute_nback(pid)
    nback_rows.append(nb)

    cpt_row = {"pid": pid}
    for sess in ['1', '2']:
        cpt_row.update(compute_cpt(pid, sess))
    cpt_rows.append(cpt_row)

    mem_row = {"pid": pid}
    for sess in ['1', '2']:
        mem_row.update(compute_cptmem(pid, sess))
    cptmem_rows.append(mem_row)

    print("done")

# ── save individual tables ────────────────────────────────────────────────────
nback_df  = pd.DataFrame(nback_rows).set_index('pid')
cpt_df    = pd.DataFrame(cpt_rows).set_index('pid')
cptmem_df = pd.DataFrame(cptmem_rows).set_index('pid')

nback_df.to_csv(os.path.join(OUT_DIR, "nback_metrics.csv"))
cpt_df.to_csv(os.path.join(OUT_DIR, "cpt_metrics.csv"))
cptmem_df.to_csv(os.path.join(OUT_DIR, "cptmem_metrics.csv"))

# ── combined table ────────────────────────────────────────────────────────────
combined = pd.concat([nback_df, cpt_df, cptmem_df], axis=1)
combined.to_csv(os.path.join(OUT_DIR, "all_metrics_combined.csv"))

print(f"\nDone. Files written to {OUT_DIR}:")
print("  nback_metrics.csv")
print("  cpt_metrics.csv")
print("  cptmem_metrics.csv")
print("  all_metrics_combined.csv")
