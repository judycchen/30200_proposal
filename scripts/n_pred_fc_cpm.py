"""
n_pred_fc_cpm.py
----------------
Aim 2: Does predicted fMRI FC (derived from task fNIRS) correlate with
individual behavioral performance?

Three-stage analysis:

  Stage A — Functional connectivity (FC) from predicted task fMRI.
    Static FC = Pearson correlation across TRs, per subject, Fisher z-transformed.
    Run for each task in TASKS.

  Stage B — Connectome-based Predictive Modeling (CPM; Shen et al. 2017).
    LOO cross-validation across subjects:
      1. Edge selection: edges correlated with behavior at p < EDGE_THRESH_P
         (separately for positive and negative networks) in the training set.
      2. Network strength: sum of selected edges per subject.
      3. Linear model: behavior ~ pos_strength + neg_strength (+ intercept).
      4. Predict held-out subject.
    Permutation test (N_PERMS shuffles of behavior) to assess significance.

  Stage C — Real fMRI CPT benchmark (AVA_RSA dataset).
    Extract 122-ROI time series from AVA_RSA vCPT BOLD (same atlas space),
    compute group-mean FC, and correlate edge-to-edge with predicted CPT2
    group-mean FC. Tests whether aPCR-predicted FC recovers the real CPT FC
    structure measured by fMRI in an independent dataset.

Behavioral targets (from k_parse_timing_behavior.py):
  Nback : nback_acc_2back  AND  nback_dprime_2back
  CPT2  : cpt2_dprime      AND  cpt2_acc

Atlas note: CPM is run on the 122-ROI Yeo/Brainnetome atlas. This is a
data-driven CPM, not a replication of the published saCPM (Rosenberg 268-ROI
Shen atlas). Results are exploratory given N ~20.

Inputs:
  - data/l_task-pred-fmri/{Task}_pred_SxRxT.mat   (n_subj, 122, 300)
  - data/k_task-timing/behavioral_scores.csv
  - /project/mdrosenberg/AC/AVA_RSA/data/BOLD_ts/avCPT_visual/  (Stage C)
  - data/a_fmri-roi-ts/atlas_122_resampled.nii.gz               (Stage C)

Outputs (data/n_cpm/):
  - {Task}_pred_fc_SxRxR.mat      : predicted FC matrices (n_subj, 122, 122)
  - cpm_results.mat               : r, p (parametric + permutation) per task/target
  - cpm_null_dist.mat             : permutation null distributions
  - C_ava_cpt_fc_SxRxR.mat        : real vCPT FC matrices (n_ava, 122, 122)
  - C_fc_comparison.mat           : group-mean FCs + edge r/p
  results/n_cpm/:
  - cpm_scatter_{Task}_{target}.png : predicted vs actual scatter plots
  - C_fc_comparison.png           : side-by-side FC matrices + edge scatter

Usage:
  python scripts/n_pred_fc_cpm.py
"""

import os
import glob
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker

np.random.seed(86)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = "/project/ycleong/users/judycchen/prediction-proj"
PRED_DIR    = os.path.join(BASE, "data/l_task-pred-fmri")
BEHAV_FILE  = os.path.join(BASE, "data/k_task-timing/behavioral_scores.csv")
OUT_DIR     = os.path.join(BASE, "data/n_cpm")
FIG_DIR     = os.path.join(BASE, "results/n_cpm")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── params ────────────────────────────────────────────────────────────────────
AVA_CPT_DIR = "/project/mdrosenberg/AC/AVA_RSA/data/BOLD_ts/avCPT_visual"
ATLAS_FILE  = os.path.join(BASE, "data/a_fmri-roi-ts/atlas_122_resampled.nii.gz")

TASKS = ['Nback', 'CPT2']
BEHAV_TARGETS = {
    'Nback': ['nback_acc_2back', 'nback_dprime_2back'],
    'CPT2':  ['cpt2_dprime',     'cpt2_acc'],
}
EDGE_THRESH_P = 0.01
N_PERMS       = 1000
N_ROIS        = 122
N_EDGES       = N_ROIS * (N_ROIS - 1) // 2   # upper triangle = 7381

# ── helpers ───────────────────────────────────────────────────────────────────

def compute_fc(pred_RxT):
    """Static FC: Pearson correlation across TRs. Returns (122, 122)."""
    return np.corrcoef(pred_RxT)

def fisher_z(fc_RxR):
    """Fisher r-to-z transform, clipped to avoid ±inf."""
    return np.arctanh(np.clip(fc_RxR, -0.999, 0.999))

def extract_upper_tri(fc_RxR, n_rois=N_ROIS):
    """Extract upper triangle (k=1) as a flat vector of length N_EDGES."""
    idx = np.triu_indices(n_rois, k=1)
    return fc_RxR[idx]

def edge_pearsonr_vectorized(fc_SxE, behavior_S):
    """
    Compute Pearson r and p between each edge and behavior across training subjects.
    fc_SxE   : (n_train, N_EDGES)
    behavior_S: (n_train,)
    Returns r_E, p_E each of shape (N_EDGES,).
    """
    n = len(behavior_S)
    beh_z    = (behavior_S - behavior_S.mean()) / (behavior_S.std() + 1e-12)
    fc_z     = (fc_SxE - fc_SxE.mean(axis=0)) / (fc_SxE.std(axis=0) + 1e-12)
    r_E      = (fc_z * beh_z[:, None]).mean(axis=0)
    r_E      = np.clip(r_E, -0.9999, 0.9999)
    # two-tailed p from t-distribution
    t_E      = r_E * np.sqrt((n - 2) / (1 - r_E**2 + 1e-12))
    from scipy.stats import t as t_dist
    p_E      = 2 * t_dist.sf(np.abs(t_E), df=n - 2)
    return r_E, p_E

def run_cpm(fc_SxE, behavior_S):
    """
    LOO CPM. Returns:
      pred_behavior : (n_subj,) predicted behavioral scores
      r_observed    : Pearson r between pred and actual
      p_observed    : parametric p-value
    """
    n_subj = len(behavior_S)
    pred_behavior = np.full(n_subj, np.nan)

    for loo in range(n_subj):
        train_idx  = [i for i in range(n_subj) if i != loo]
        train_fc   = fc_SxE[train_idx]           # (n-1, N_EDGES)
        train_beh  = behavior_S[train_idx]        # (n-1,)
        test_fc    = fc_SxE[loo]                  # (N_EDGES,)

        # edge selection in training set
        r_edge, p_edge = edge_pearsonr_vectorized(train_fc, train_beh)
        pos_mask = (p_edge < EDGE_THRESH_P) & (r_edge > 0)
        neg_mask = (p_edge < EDGE_THRESH_P) & (r_edge < 0)

        # network strength (sum of selected edges)
        pos_str_train = train_fc[:, pos_mask].sum(axis=1) if pos_mask.any() else np.zeros(len(train_idx))
        neg_str_train = train_fc[:, neg_mask].sum(axis=1) if neg_mask.any() else np.zeros(len(train_idx))
        pos_str_test  = float(test_fc[pos_mask].sum()) if pos_mask.any() else 0.0
        neg_str_test  = float(test_fc[neg_mask].sum()) if neg_mask.any() else 0.0

        # linear model: behavior ~ pos_str + neg_str + intercept (OLS)
        X_train = np.column_stack([pos_str_train, neg_str_train, np.ones(len(train_idx))])
        coef, _, _, _ = np.linalg.lstsq(X_train, train_beh, rcond=None)

        pred_behavior[loo] = np.dot([pos_str_test, neg_str_test, 1.0], coef)

    valid = np.isfinite(pred_behavior) & np.isfinite(behavior_S)
    if valid.sum() < 3:
        return pred_behavior, np.nan, np.nan

    r_obs, p_obs = stats.pearsonr(pred_behavior[valid], behavior_S[valid])
    return pred_behavior, float(r_obs), float(p_obs)

# ── load behavioral scores ────────────────────────────────────────────────────
print("=" * 60)
print("Loading behavioral scores")
print("=" * 60)

behav_df = pd.read_csv(BEHAV_FILE)
print(behav_df.to_string(index=False))

# ── STAGE A + B: loop over tasks and behavioral targets ──────────────────────
all_results  = {}
all_null     = {}

for task in TASKS:
    pred_path = os.path.join(PRED_DIR, f'{task}_pred_SxRxT.mat')
    if not os.path.exists(pred_path):
        print(f"\n{task}: predicted fMRI file not found — skipping")
        continue

    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    pred_mat   = sio.loadmat(pred_path)
    pred_SxRxT = pred_mat['pred_SxRxT']   # (n_subj, 122, 300)
    print(f"Predicted fMRI: {pred_SxRxT.shape}")

    # load subject IDs saved by l_apply_model_tasks.py
    subj_ids_path = os.path.join(PRED_DIR, f'{task}_subj_ids.json')
    if os.path.exists(subj_ids_path):
        import json as _json
        with open(subj_ids_path) as _f:
            pred_subj_ids = _json.load(_f)
    else:
        pred_subj_ids = [f's{i+1:02d}' for i in range(pred_SxRxT.shape[0])]
        print(f"  WARNING: {task}_subj_ids.json not found — run l_ first")
    print(f"  Subject IDs: {pred_subj_ids}")
    n_subj = len(pred_subj_ids)

    # ── Stage A: compute FC matrices ─────────────────────────────────────────
    print(f"\n── Stage A: FC matrices ──")
    fc_SxRxR = np.zeros((n_subj, N_ROIS, N_ROIS))
    for s in range(n_subj):
        pred_RxT     = pred_SxRxT[s]                  # (122, 300)
        fc_SxRxR[s]  = fisher_z(compute_fc(pred_RxT))

    fc_mean_diag_off = float(np.nanmean(np.abs(
        fc_SxRxR[:, np.triu_indices(N_ROIS, k=1)[0], np.triu_indices(N_ROIS, k=1)[1]]
    )))
    print(f"Mean |z-FC| (upper tri): {fc_mean_diag_off:.4f}")

    fc_mat_path = os.path.join(OUT_DIR, f'{task}_pred_fc_SxRxR.mat')
    sio.savemat(fc_mat_path, {'fc_SxRxR': fc_SxRxR, 'subj_ids': pred_subj_ids})
    print(f"Saved {task}_pred_fc_SxRxR.mat")

    # edge matrix: (n_subj, N_EDGES)
    fc_SxE = np.array([extract_upper_tri(fc_SxRxR[s]) for s in range(n_subj)])

    # ── Stage B: CPM per behavioral target ───────────────────────────────────
    # align by subj_id: inner-join predicted FC subjects with behavioral scores
    behav_df_indexed = behav_df.set_index('subj_id') if 'subj_id' in behav_df.columns else behav_df

    for target in BEHAV_TARGETS[task]:
        print(f"\n── CPM: {task} ~ {target} ──")

        if target not in behav_df.columns:
            print(f"  '{target}' not in behavioral_scores.csv — skipping")
            continue

        # match subjects: keep only those with both predicted FC and behavioral score
        valid_subjs = [sid for sid in pred_subj_ids
                       if sid in behav_df_indexed.index
                       and pd.notna(behav_df_indexed.loc[sid, target])]
        valid_fc_idx  = [pred_subj_ids.index(sid) for sid in valid_subjs]

        behavior_S = behav_df_indexed.loc[valid_subjs, target].values.astype(float)
        fc_aligned = fc_SxE[valid_fc_idx]

        print(f"  N subjects with {target}: {len(behavior_S)}")
        print(f"  Behavior — mean={behavior_S.mean():.3f}, std={behavior_S.std():.3f}, "
              f"range=[{behavior_S.min():.3f}, {behavior_S.max():.3f}]")

        # observed CPM
        pred_beh, r_obs, p_obs = run_cpm(fc_aligned, behavior_S)
        print(f"  Observed: r={r_obs:.4f}, parametric p={p_obs:.4f}")

        # permutation test
        null_r = np.zeros(N_PERMS)
        rng    = np.random.default_rng(86)
        for perm in range(N_PERMS):
            beh_shuffled = rng.permutation(behavior_S)
            _, r_perm, _ = run_cpm(fc_aligned, beh_shuffled)
            null_r[perm] = r_perm if np.isfinite(r_perm) else 0.0
            if (perm + 1) % 200 == 0:
                print(f"    permutation {perm+1}/{N_PERMS}...")

        p_perm = float(np.mean(null_r >= r_obs)) if np.isfinite(r_obs) else np.nan
        print(f"  Permutation p={p_perm:.4f}  (r_obs={r_obs:.4f} vs null "
              f"mean={null_r.mean():.4f}, std={null_r.std():.4f})")

        key = f'{task}_{target}'
        all_results[key] = {
            'task': task, 'target': target, 'n_subj': len(behavior_S),
            'r_obs': r_obs, 'p_parametric': p_obs, 'p_permutation': p_perm,
            'pred_behavior': pred_beh, 'true_behavior': behavior_S,
        }
        all_null[key] = null_r

        # scatter plot
        fig, ax = plt.subplots(figsize=(5, 5))
        valid_pts = np.isfinite(pred_beh) & np.isfinite(behavior_S)
        ax.scatter(behavior_S[valid_pts], pred_beh[valid_pts], color='#2166ac',
                   alpha=0.8, s=60, edgecolors='white', linewidths=0.5)

        # regression line
        if valid_pts.sum() > 2:
            m, b = np.polyfit(behavior_S[valid_pts], pred_beh[valid_pts], 1)
            xline = np.linspace(behavior_S[valid_pts].min(), behavior_S[valid_pts].max(), 50)
            ax.plot(xline, m * xline + b, color='#b2182b', linewidth=1.5)

        ax.set_xlabel(f'Actual {target}', fontsize=12)
        ax.set_ylabel('CPM predicted', fontsize=12)
        ax.set_title(
            f'{task} ~ {target}\n'
            f'r={r_obs:.3f}, parametric p={p_obs:.3f}, perm p={p_perm:.3f}\n'
            f'N={valid_pts.sum()} | CAUTION: exploratory, N~20',
            fontsize=10
        )
        plt.tight_layout()
        fig_name = f'cpm_scatter_{task}_{target}.png'
        fig.savefig(os.path.join(FIG_DIR, fig_name), dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_name}")

# ── STAGE C: AVA_RSA real fMRI vCPT — FC benchmark ───────────────────────────
print("\n" + "=" * 60)
print("STAGE C: AVA_RSA real fMRI vCPT — FC benchmark")
print("=" * 60)

cpt2_fc_path = os.path.join(OUT_DIR, 'CPT2_pred_fc_SxRxR.mat')
if not os.path.exists(cpt2_fc_path):
    print("CPT2 predicted FC not found — skipping Stage C (run with CPT2 in TASKS first)")
elif not os.path.exists(ATLAS_FILE):
    print(f"Atlas not found: {ATLAS_FILE} — skipping Stage C")
else:
    # load predicted CPT2 group-mean FC
    cpt2_fc_mat   = sio.loadmat(cpt2_fc_path)
    cpt2_fc_SxRxR = cpt2_fc_mat['fc_SxRxR']           # (n_pred, 122, 122)
    cpt2_fc_mean  = cpt2_fc_SxRxR.mean(axis=0)        # (122, 122)
    n_pred_subj   = cpt2_fc_SxRxR.shape[0]
    print(f"Predicted CPT2 FC loaded: {cpt2_fc_SxRxR.shape}  (mean over {n_pred_subj} subjects)")

    # set up atlas masker — no resampling: atlas and AVA_RSA data share the same space
    masker = NiftiLabelsMasker(
        labels_img=ATLAS_FILE,
        standardize=False,
        resampling_target=None,
        memory_level=0,
    )
    masker.fit()

    # find all AVA_RSA vCPT NIfTI files
    nii_files = sorted(glob.glob(os.path.join(AVA_CPT_DIR, 'errts.*.vCPT.01.tproject.nii')))
    print(f"Found {len(nii_files)} AVA_RSA vCPT NIfTI files")

    ava_fc_list = []
    ava_ids     = []
    for fpath in nii_files:
        basename = os.path.basename(fpath)
        subj_id  = basename.split('.')[1]   # e.g. '1001'
        try:
            roi_ts = masker.transform(fpath)   # (T, 122)
            fc     = fisher_z(compute_fc(roi_ts.T))   # (122, 122); compute_fc expects (122, T)
            ava_fc_list.append(fc)
            ava_ids.append(subj_id)
        except Exception as e:
            print(f"  WARNING: subject {subj_id} failed — {e}")

    print(f"Successfully processed: {len(ava_ids)} AVA_RSA subjects")

    ava_fc_SxRxR = np.stack(ava_fc_list, axis=0)   # (n_ava, 122, 122)
    ava_fc_mean  = ava_fc_SxRxR.mean(axis=0)       # (122, 122)

    # edge-level correlation between group means
    ava_tri  = extract_upper_tri(ava_fc_mean)
    cpt2_tri = extract_upper_tri(cpt2_fc_mean)
    r_edge, p_edge = stats.pearsonr(ava_tri, cpt2_tri)
    print(f"\nEdge-level FC similarity: r={r_edge:.4f}, p={p_edge:.2e}")
    print(f"  Predicted CPT2 N={n_pred_subj}  |  Real vCPT (AVA_RSA) N={len(ava_ids)}")

    # save
    sio.savemat(
        os.path.join(OUT_DIR, 'C_ava_cpt_fc_SxRxR.mat'),
        {'fc_SxRxR': ava_fc_SxRxR, 'subj_ids': ava_ids}
    )
    sio.savemat(
        os.path.join(OUT_DIR, 'C_fc_comparison.mat'),
        {'ava_fc_mean': ava_fc_mean, 'pred_cpt2_fc_mean': cpt2_fc_mean,
         'r_edge': r_edge, 'p_edge': p_edge,
         'n_ava': len(ava_ids), 'n_pred': n_pred_subj}
    )
    print("Saved C_ava_cpt_fc_SxRxR.mat and C_fc_comparison.mat")

    # figure: side-by-side FC matrices + edge scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vlim = np.percentile(np.abs(np.concatenate([cpt2_tri, ava_tri])), 95)

    im0 = axes[0].imshow(cpt2_fc_mean, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                         interpolation='nearest')
    axes[0].set_title(f'Predicted CPT2 FC\n(N={n_pred_subj} fNIRS→fMRI)', fontsize=11)
    axes[0].set_xlabel('ROI'); axes[0].set_ylabel('ROI')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ava_fc_mean, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                         interpolation='nearest')
    axes[1].set_title(f'Real vCPT FC (AVA_RSA)\n(N={len(ava_ids)} fMRI)', fontsize=11)
    axes[1].set_xlabel('ROI')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # edge scatter
    axes[2].scatter(ava_tri, cpt2_tri, alpha=0.03, s=1, color='#2166ac', rasterized=True)
    m, b = np.polyfit(ava_tri, cpt2_tri, 1)
    xline = np.linspace(ava_tri.min(), ava_tri.max(), 100)
    axes[2].plot(xline, m * xline + b, color='#b2182b', linewidth=1.5)
    axes[2].set_xlabel('Real vCPT edge z (AVA_RSA)', fontsize=11)
    axes[2].set_ylabel('Predicted CPT2 edge z', fontsize=11)
    axes[2].set_title(f'Edge-level FC similarity\nr={r_edge:.3f}, p={p_edge:.1e}\n'
                      f'N_edges={N_EDGES}', fontsize=11)

    plt.suptitle('Stage C: Predicted CPT2 vs Real vCPT (AVA_RSA) — 122-ROI FC',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'C_fc_comparison.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved C_fc_comparison.png")

# ── save consolidated results ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving consolidated results")
print("=" * 60)

# results summary CSV
rows = []
for key, res in all_results.items():
    rows.append({
        'task':          res['task'],
        'behav_target':  res['target'],
        'n_subj':        res['n_subj'],
        'r_observed':    round(res['r_obs'], 4) if np.isfinite(res['r_obs']) else np.nan,
        'p_parametric':  round(res['p_parametric'], 4) if np.isfinite(res['p_parametric']) else np.nan,
        'p_permutation': round(res['p_permutation'], 4) if np.isfinite(res['p_permutation']) else np.nan,
    })
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(OUT_DIR, 'cpm_results_summary.csv'), index=False)
print(results_df.to_string(index=False))

# detailed .mat with predicted vs actual and null distributions
mat_results = {}
for key, res in all_results.items():
    safe_key = key.replace('-', '_')
    mat_results[f'{safe_key}_r_obs']          = res['r_obs']
    mat_results[f'{safe_key}_p_parametric']   = res['p_parametric']
    mat_results[f'{safe_key}_p_permutation']  = res['p_permutation']
    mat_results[f'{safe_key}_pred_behavior']  = res['pred_behavior']
    mat_results[f'{safe_key}_true_behavior']  = res['true_behavior']

mat_null = {k.replace('-', '_'): v for k, v in all_null.items()}

sio.savemat(os.path.join(OUT_DIR, 'cpm_results.mat'),  mat_results)
sio.savemat(os.path.join(OUT_DIR, 'cpm_null_dist.mat'), mat_null)
print("Saved cpm_results.mat, cpm_null_dist.mat")

print(f"\nAll outputs in: {OUT_DIR}  |  Figures in: {FIG_DIR}")
print("Done: n_pred_fc_cpm.py")
