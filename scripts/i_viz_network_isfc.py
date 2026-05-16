"""
i_viz_network_isfc.py
---------------------
Phase 5: Python visualizations — Figure 3b + Figure 4

Produces:
  Figure 3b : Network bar plot — median r per network, individual ROI dots,
              fraction significant above each bar
  Figure 4  : ISFC heatmaps — observed BOLD (left) vs predicted BOLD (right),
              ROIs ordered by network, colormap -0.3 to 0.3

Inputs:
  - data/d_apcr-model/apcr_significance.mat
  - data/d_apcr-model/apcr_isfc.mat
  - Yeo_17N_114_Brainnetome_subcortical_8.csv

Outputs (data/e_visualization/):
  - network/fig3b_network_barplot.jpg
  - network/network_summary.csv
  - isfc/fig4_isfc_heatmap.jpg

Usage:
  python scripts/i_viz_network_isfc.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(86)

# ── variant flags ─────────────────────────────────────────────────────────────
# Must match the settings used in e_apcr_model.py / g_apcr_isfc.py.
# Set PCA75 = True to visualize the 75%-variance PCA run (d_apcr-model_pca75).
# Set FRONTAL_ONLY = True to visualize the frontal-channels-only run.
# Both flags can be combined.

FRONTAL_ONLY = False
PCA75        = True

# ── paths ─────────────────────────────────────────────────────────────────────

BASE   = "/project/ycleong/users/judycchen/prediction-proj"
MASKS  = ("/project/ycleong/users/judycchen/fnirs_fmri_models/data/sherlock"
          "/fmri/masks/Yeo_17N_114_Brainnetome_subcortical_8")

_out_suffix = ("_frontal" if FRONTAL_ONLY else "") + ("_pca75" if PCA75 else "")
MODEL_DIR = os.path.join(BASE, f"data/d_apcr-model{_out_suffix}")
NET_DIR   = os.path.join(BASE, f"results/e_visualization{_out_suffix}/network")
ISFC_DIR  = os.path.join(BASE, f"results/e_visualization{_out_suffix}/isfc")
os.makedirs(NET_DIR,  exist_ok=True)
os.makedirs(ISFC_DIR, exist_ok=True)

# ── shared: load ROI labels and assign networks ───────────────────────────────

print("Loading ROI labels...")
labels_df = pd.read_csv(
    os.path.join(MASKS, "Yeo_17N_114_Brainnetome_subcortical_8.csv"),
    header=None,
    names=['roi_id', 'full_name', 'short_name', 'R', 'G', 'B', 'network_code']
)

def get_network(short_name):
    if 'Subcortical' in short_name: return 'SUBC'
    if 'Vis'         in short_name: return 'VIS'
    if 'SomMot'      in short_name: return 'SM'
    if 'DorsAttn'    in short_name: return 'DAN'
    if 'SalVentAttn' in short_name: return 'VAN'
    if 'Limbic'      in short_name: return 'LIMB'
    if 'Cont'        in short_name: return 'CONT'
    if 'Default'     in short_name: return 'DMN'
    if 'TempPar'     in short_name: return 'DMN'
    return 'OTHER'

labels_df['network'] = labels_df['short_name'].apply(get_network)

NETWORK_ORDER = ['VIS', 'SM', 'DAN', 'VAN', 'LIMB', 'CONT', 'DMN', 'SUBC']

# ── load model significance results ──────────────────────────────────────────

print("Loading significance results...")
sig_data  = sio.loadmat(os.path.join(MODEL_DIR, "apcr_significance.mat"))
grp_grp_r = sig_data['grp_grp_corr_Rx1'].flatten()
sigmask   = sig_data['grp_grp_corr_sigmask_Rx1'].flatten().astype(bool)

labels_df['r']   = grp_grp_r
labels_df['sig'] = sigmask
print(f"ROIs: {len(grp_grp_r)} | Significant: {sigmask.sum()}")

# ── load ISFC results ─────────────────────────────────────────────────────────

print("Loading ISFC results...")
isfc_data   = sio.loadmat(os.path.join(MODEL_DIR, "apcr_isfc.mat"))
true_isfc   = isfc_data['true_isfc_RxR']            # (122, 122) raw asymmetric mean ISFC
pred_isfc   = isfc_data['pred_isfc_RxR']            # (122, 122) raw asymmetric mean ISFC
isfc_r      = float(isfc_data['isfc_similarity'])
isfc_p      = float(isfc_data['isfc_pval'])
print(f"ISFC matrix similarity: r={isfc_r:.3f}, p={isfc_p:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3b: NETWORK BAR PLOT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Figure 3b: Network bar plot")
print("=" * 60)

df = labels_df[labels_df['network'].isin(NETWORK_ORDER)].copy()
df['network'] = pd.Categorical(df['network'], categories=NETWORK_ORDER, ordered=True)

summary = df.groupby('network', observed=True).apply(
    lambda x: pd.Series({
        'median_r':   float(np.median(x['r'])),
        'n_sig':      int(x['sig'].sum()),
        'n_total':    len(x),
        'prop_label': f"{int(x['sig'].sum())}/{len(x)}"
    }), include_groups=False
).reset_index()

print(summary[['network', 'median_r', 'n_sig', 'n_total']].to_string(index=False))

fig3b, ax = plt.subplots(figsize=(8, 5))
x_pos     = np.arange(len(NETWORK_ORDER))
rng       = np.random.default_rng(86)

# bars
ax.bar(x_pos, summary['median_r'], width=0.45,
       color='#F74949', edgecolor='#F74949', linewidth=1.5)

# individual ROI dots with jitter
for i, net in enumerate(NETWORK_ORDER):
    roi_r  = df[df['network'] == net]['r'].values
    jitter = rng.uniform(-0.12, 0.12, size=len(roi_r))
    ax.scatter(i + jitter, roi_r,
               color='grey', s=12, alpha=0.7, linewidths=0.5, zorder=3)

# fraction significant above bars
for i, row in summary.iterrows():
    y_label = max(float(row['median_r']), 0) + 0.025
    ax.text(x_pos[i], y_label, row['prop_label'],
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(NETWORK_ORDER, fontsize=12, fontweight='bold')
ax.set_ylabel("Pearson r (predicted vs observed fMRI)", fontsize=11)
ax.set_title("NNW fNIRS→fMRI aPCR — predictive accuracy by network", fontsize=11)
ax.set_ylim(-0.22, 0.48)
ax.set_yticks(np.arange(-0.2, 0.5, 0.2))
ax.yaxis.set_tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig3b_path = os.path.join(NET_DIR, "fig3b_network_barplot.jpg")
plt.savefig(fig3b_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3b_path}")

summary.to_csv(os.path.join(NET_DIR, "network_summary.csv"), index=False)
print("Saved: network_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: ISFC HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Figure 4: ISFC heatmaps")
print("=" * 60)

# Sort ROIs by network order
labels_sorted = labels_df[labels_df['network'].isin(NETWORK_ORDER)].copy()
labels_sorted['network'] = pd.Categorical(
    labels_sorted['network'], categories=NETWORK_ORDER, ordered=True)
labels_sorted = labels_sorted.sort_values('network').reset_index(drop=True)
roi_order = labels_sorted['roi_id'].values - 1   # 0-indexed

# Reorder the raw asymmetric matrices directly — no symmetrization
true_reordered = true_isfc[np.ix_(roi_order, roi_order)]
pred_reordered = pred_isfc[np.ix_(roi_order, roi_order)]

# Network boundaries and midpoints for axis labels
network_sizes      = labels_sorted.groupby('network', observed=True).size()
network_boundaries = np.cumsum(network_sizes.values)[:-1]
bounds_ext         = np.concatenate([[0], np.cumsum(network_sizes.values)])
network_midpoints  = (bounds_ext[:-1] + bounds_ext[1:]) / 2

# Colormap matching Gao et al. — smooth diverging, 0 exactly centered
# RdBu_r: blue=negative, white=zero, red=positive, smooth transitions
cmap_bwr = plt.cm.RdBu_r

# Shared ±0.3 scale for both panels, matching the reference figure.
# Symmetric around exactly 0.0 so white = no connectivity.
CLIM = 0.3
CTICKS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, matrix, title in zip(
    axes,
    [true_reordered, pred_reordered],
    ['Observed BOLD ISFC', 'Predicted BOLD ISFC']
):
    im = ax.imshow(matrix, cmap=cmap_bwr, vmin=-CLIM, vmax=CLIM,
                   aspect='auto', interpolation='none')

    for b in network_boundaries:
        ax.axhline(b - 0.5, color='black', linewidth=0.8)
        ax.axvline(b - 0.5, color='black', linewidth=0.8)

    ax.set_xticks(network_midpoints)
    ax.set_xticklabels(NETWORK_ORDER, fontsize=9, fontweight='bold')
    ax.set_yticks(network_midpoints)
    ax.set_yticklabels(NETWORK_ORDER, fontsize=9, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('r', fontsize=10)
    cbar.set_ticks(CTICKS)
    cbar.ax.tick_params(labelsize=8)

fig4.text(0.5, 0.02,
          f"Matrix similarity: Pearson r = {isfc_r:.3f}, p = {isfc_p:.3f}",
          ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig4_path = os.path.join(ISFC_DIR, "fig4_isfc_heatmap.jpg")
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path}")

print("\nDone: i_viz_network_isfc.py")
print(f"Outputs:")
print(f"  {fig3b_path}")
print(f"  {fig4_path}")
