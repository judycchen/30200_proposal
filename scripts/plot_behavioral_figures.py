"""
plot_behavioral_figures.py
---------------------------
Generates two figures for the MACS 30200 proposal:

  fig5_manip_check_nback.jpg
    Paired plot of 1-back vs 2-back accuracy (N=28 analytic sample).
    Fulfills the manipulation check stated in the Methods section.

  fig6_behavioral_overview.jpg
    3-panel figure showing individual-differences distributions for the
    three behavioral constructs targeted in Phase 5:
      A. N-back iSD-RT by condition (primary attentional stability DV)
      B. CPT commission error rate across sessions (sustained attention DV)
      C. CPT-MEM face d-prime across sessions (incidental encoding DV)

Outputs saved to:
  /project/ycleong/users/judycchen/prediction-proj/macs_30200_overleaf-zip/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── config ────────────────────────────────────────────────────────────────────
DATA    = "/project/ycleong/datasets/CogTasks-NNW_judy/psychopy_data/behavioral_results"
OUT_DIR = "/project/ycleong/users/judycchen/prediction-proj/macs_30200_overleaf-zip/figures"
os.makedirs(OUT_DIR, exist_ok=True)

EXCL    = {5, 13, 14, 15, 26, 27}   # fNIRS-excluded participants

GREY    = '#555555'
BLUE    = '#2166AC'
RED     = '#D6604D'
PURPLE  = '#762A83'
LINE_C  = '#AAAAAA'
DOT_C   = '#333333'

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# ── load and filter to N=28 ───────────────────────────────────────────────────
nb  = pd.read_csv(f'{DATA}/nback_metrics.csv',  index_col='pid')
cpt = pd.read_csv(f'{DATA}/cpt_metrics.csv',    index_col='pid')
mem = pd.read_csv(f'{DATA}/cptmem_metrics.csv', index_col='pid')

nb  = nb[~nb.index.isin(EXCL)].copy()
cpt = cpt[~cpt.index.isin(EXCL)].copy()
mem = mem[~mem.index.isin(EXCL)].copy()

print(f"Analytic sample: N={len(nb)}")

# ── helper: jitter ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def jitter(n, width=0.08):
    return rng.uniform(-width, width, size=n)

def add_sig_bar(ax, x1, x2, y, h, p):
    """Draw a significance bar between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color=GREY)
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = 'n.s.'
    ax.text((x1+x2)/2, y+h*1.1, label, ha='center', va='bottom', fontsize=11, color=GREY)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: MANIPULATION CHECK
# ══════════════════════════════════════════════════════════════════════════════

acc_1b = nb['nback_1back_accuracy'].dropna()
acc_2b = nb['nback_2back_accuracy'].dropna()

# Paired t-test on participants with both conditions
both = nb[['nback_1back_accuracy','nback_2back_accuracy']].dropna()
t_stat, p_val = stats.ttest_rel(both['nback_1back_accuracy'], both['nback_2back_accuracy'])
d_cohen = (both['nback_1back_accuracy'] - both['nback_2back_accuracy']).mean() / \
          (both['nback_1back_accuracy'] - both['nback_2back_accuracy']).std(ddof=1)

print(f"\nManip check: t({len(both)-1}) = {t_stat:.2f}, p = {p_val:.4f}, d = {d_cohen:.2f}")
print(f"  1-back: M={acc_1b.mean():.3f}, SD={acc_1b.std():.3f}")
print(f"  2-back: M={acc_2b.mean():.3f}, SD={acc_2b.std():.3f}")

fig5, ax = plt.subplots(figsize=(3.8, 4.8))

xs = [0, 1]
colors = [BLUE, RED]
labels = ['1-back', '2-back']
data   = [acc_1b.values, acc_2b.values]

# Violin
vp = ax.violinplot(data, positions=xs, widths=0.4, showmedians=False, showextrema=False)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors[i])
    body.set_alpha(0.25)
    body.set_edgecolor(colors[i])
    body.set_linewidth(0.8)

# Box (IQR)
for i, (x, vals) in enumerate(zip(xs, data)):
    q25, q75 = np.percentile(vals, [25, 75])
    med      = np.median(vals)
    ax.plot([x, x], [q25, q75], lw=2.5, color=colors[i], solid_capstyle='round', zorder=3)
    ax.scatter(x, med, s=45, color=colors[i], zorder=4, linewidth=0)

# Connecting lines for paired participants
for idx in both.index:
    y1 = both.loc[idx, 'nback_1back_accuracy']
    y2 = both.loc[idx, 'nback_2back_accuracy']
    ax.plot([0, 1], [y1, y2], color=LINE_C, lw=0.7, alpha=0.6, zorder=1)

# Scatter dots
for i, (x, vals) in enumerate(zip(xs, data)):
    ax.scatter(x + jitter(len(vals)), vals,
               color=colors[i], s=22, alpha=0.75, linewidths=0.4,
               edgecolors='white', zorder=2)

# Significance bar
top = max(acc_1b.max(), acc_2b.max())
add_sig_bar(ax, 0, 1, top + 0.03, 0.025, p_val)

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Accuracy (proportion correct)', fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_xlim(-0.5, 1.5)
ax.set_title('N-back manipulation check', fontsize=12, fontweight='bold', pad=8)

# Stats annotation inside plot
ax.text(0.5, 0.04,
        f't({len(both)-1}) = {t_stat:.2f}, p {"< .001" if p_val < .001 else f"= {p_val:.3f}"},\nCohen\'s d = {d_cohen:.2f}',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color=GREY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', linewidth=0.6))

plt.tight_layout()
path5 = os.path.join(OUT_DIR, 'fig5_manip_check_nback.jpg')
plt.savefig(path5, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {path5}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: BEHAVIORAL OVERVIEW (3 panels)
# ══════════════════════════════════════════════════════════════════════════════

fig6, axes = plt.subplots(1, 3, figsize=(11, 4.5))

# ── Panel A: N-back iSD-RT ────────────────────────────────────────────────────
ax = axes[0]
isd_1b = nb['nback_1back_rt_isd'].dropna() * 1000   # convert to ms
isd_2b = nb['nback_2back_rt_isd'].dropna() * 1000

both_isd = nb[['nback_1back_rt_isd','nback_2back_rt_isd']].dropna() * 1000
t_isd, p_isd = stats.ttest_rel(both_isd['nback_1back_rt_isd'], both_isd['nback_2back_rt_isd'])
d_isd = (both_isd['nback_1back_rt_isd'] - both_isd['nback_2back_rt_isd']).mean() / \
        (both_isd['nback_1back_rt_isd'] - both_isd['nback_2back_rt_isd']).std(ddof=1)

print(f"\nN-back iSD-RT: t({len(both_isd)-1}) = {t_isd:.2f}, p = {p_isd:.4f}, d = {d_isd:.2f}")
print(f"  1-back: M={isd_1b.mean():.1f}ms, SD={isd_1b.std():.1f}ms")
print(f"  2-back: M={isd_2b.mean():.1f}ms, SD={isd_2b.std():.1f}ms")

for x, vals, col in zip([0, 1], [isd_1b.values, isd_2b.values], [BLUE, RED]):
    vp = ax.violinplot([vals], positions=[x], widths=0.4, showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(col); body.set_alpha(0.25); body.set_edgecolor(col); body.set_linewidth(0.8)
    q25, q75 = np.percentile(vals, [25, 75])
    ax.plot([x, x], [q25, q75], lw=2.5, color=col, solid_capstyle='round', zorder=3)
    ax.scatter(x, np.median(vals), s=45, color=col, zorder=4, linewidth=0)
    ax.scatter(x + jitter(len(vals)), vals, color=col, s=18, alpha=0.7,
               linewidths=0.4, edgecolors='white', zorder=2)

for idx in both_isd.index:
    ax.plot([0, 1],
            [both_isd.loc[idx,'nback_1back_rt_isd'], both_isd.loc[idx,'nback_2back_rt_isd']],
            color=LINE_C, lw=0.6, alpha=0.5, zorder=1)

top_isd = max(isd_1b.max(), isd_2b.max())
add_sig_bar(ax, 0, 1, top_isd + 3, 2.5, p_isd)

ax.set_xticks([0, 1]); ax.set_xticklabels(['1-back', '2-back'], fontsize=10)
ax.set_ylabel('iSD-RT (ms)', fontsize=10)
ax.set_title('Attentional stability\n(N-back iSD-RT)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.55, 1.55)

# ── Panel B: CPT commission error rate ───────────────────────────────────────
ax = axes[1]
ce1 = cpt['cpt1_commission_rate'].dropna()
ce2 = cpt['cpt2_commission_rate'].dropna()

both_ce = cpt[['cpt1_commission_rate','cpt2_commission_rate']].dropna()
t_ce, p_ce = stats.ttest_rel(both_ce['cpt1_commission_rate'], both_ce['cpt2_commission_rate'])
r_ce, p_r_ce = stats.pearsonr(both_ce['cpt1_commission_rate'], both_ce['cpt2_commission_rate'])

print(f"\nCPT commission: session 1 vs 2: t({len(both_ce)-1}) = {t_ce:.2f}, p = {p_ce:.4f}")
print(f"  Test-retest r = {r_ce:.3f}, p = {p_r_ce:.4f}")
print(f"  S1: M={ce1.mean():.3f}, SD={ce1.std():.3f}")
print(f"  S2: M={ce2.mean():.3f}, SD={ce2.std():.3f}")

TEAL   = '#1B7837'
ORANGE = '#E08214'
for x, vals, col, lbl in zip([0, 1], [ce1.values, ce2.values], [TEAL, ORANGE], ['Session 1','Session 2']):
    vp = ax.violinplot([vals], positions=[x], widths=0.4, showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(col); body.set_alpha(0.25); body.set_edgecolor(col); body.set_linewidth(0.8)
    q25, q75 = np.percentile(vals, [25, 75])
    ax.plot([x, x], [q25, q75], lw=2.5, color=col, solid_capstyle='round', zorder=3)
    ax.scatter(x, np.median(vals), s=45, color=col, zorder=4, linewidth=0)
    ax.scatter(x + jitter(len(vals)), vals, color=col, s=18, alpha=0.7,
               linewidths=0.4, edgecolors='white', zorder=2)

for idx in both_ce.index:
    ax.plot([0, 1],
            [both_ce.loc[idx,'cpt1_commission_rate'], both_ce.loc[idx,'cpt2_commission_rate']],
            color=LINE_C, lw=0.6, alpha=0.5, zorder=1)

top_ce = max(ce1.max(), ce2.max())
# annotate test-retest instead of t-test bar
ax.text(0.5, 0.96,
        f'test–retest r = {r_ce:.2f}{"***" if p_r_ce < .001 else ("**" if p_r_ce < .01 else "*")}',
        transform=ax.transAxes, ha='center', va='top', fontsize=9, color=GREY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', linewidth=0.6))

ax.set_xticks([0, 1]); ax.set_xticklabels(['Session 1', 'Session 2'], fontsize=10)
ax.set_ylabel('Commission error rate', fontsize=10)
ax.set_title('Sustained attention\n(CPT commission errors)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.02, 1.0)

# ── Panel C: CPT-MEM face d-prime ────────────────────────────────────────────
ax = axes[2]
dp1 = mem['cptmem1_face_dprime'].dropna()
dp2 = mem['cptmem2_face_dprime'].dropna()

both_dp = mem[['cptmem1_face_dprime','cptmem2_face_dprime']].dropna()
t_dp, p_dp = stats.ttest_1samp(dp1, 0)       # test vs 0 (chance)
t_dp2, p_dp2 = stats.ttest_1samp(dp2, 0)
r_dp, p_r_dp = stats.pearsonr(both_dp['cptmem1_face_dprime'], both_dp['cptmem2_face_dprime'])

print(f"\nCPT-MEM face d': S1 vs 0: t({len(dp1)-1}) = {t_dp:.2f}, p = {p_dp:.4f}")
print(f"  S2 vs 0: t({len(dp2)-1}) = {t_dp2:.2f}, p = {p_dp2:.4f}")
print(f"  Test-retest r = {r_dp:.3f}, p = {p_r_dp:.4f}")
print(f"  S1: M={dp1.mean():.3f}, SD={dp1.std():.3f}")
print(f"  S2: M={dp2.mean():.3f}, SD={dp2.std():.3f}")

NAVY   = '#4D4DC8'
PLUM   = '#8C3FC0'
for x, vals, col in zip([0, 1], [dp1.values, dp2.values], [NAVY, PLUM]):
    vp = ax.violinplot([vals], positions=[x], widths=0.4, showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(col); body.set_alpha(0.25); body.set_edgecolor(col); body.set_linewidth(0.8)
    q25, q75 = np.percentile(vals, [25, 75])
    ax.plot([x, x], [q25, q75], lw=2.5, color=col, solid_capstyle='round', zorder=3)
    ax.scatter(x, np.median(vals), s=45, color=col, zorder=4, linewidth=0)
    ax.scatter(x + jitter(len(vals)), vals, color=col, s=18, alpha=0.7,
               linewidths=0.4, edgecolors='white', zorder=2)

for idx in both_dp.index:
    ax.plot([0, 1],
            [both_dp.loc[idx,'cptmem1_face_dprime'], both_dp.loc[idx,'cptmem2_face_dprime']],
            color=LINE_C, lw=0.6, alpha=0.5, zorder=1)

ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
ax.text(0.5, 0.96,
        f'test–retest r = {r_dp:.2f}{"*" if p_r_dp < .05 else " (n.s.)"}',
        transform=ax.transAxes, ha='center', va='top', fontsize=9, color=GREY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', linewidth=0.6))

ax.set_xticks([0, 1]); ax.set_xticklabels(['Session 1', 'Session 2'], fontsize=10)
ax.set_ylabel("Recognition sensitivity (d')", fontsize=10)
ax.set_title('Incidental encoding\n(CPT-MEM face d\')', fontsize=11, fontweight='bold')
ax.set_xlim(-0.55, 1.55)

# ── Panel labels ──────────────────────────────────────────────────────────────
for ax_i, (ax, lbl) in enumerate(zip(axes, ['A', 'B', 'C'])):
    ax.text(-0.12, 1.04, lbl, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

fig6.suptitle(f'Behavioral distributions across tasks (N = {len(nb)})',
              fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
path6 = os.path.join(OUT_DIR, 'fig6_behavioral_overview.jpg')
plt.savefig(path6, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {path6}")

print("\nAll statistics for writing the tex:")
print(f"  Manip check: 1B M={acc_1b.mean():.3f} SD={acc_1b.std():.3f}, "
      f"2B M={acc_2b.mean():.3f} SD={acc_2b.std():.3f}, "
      f"t({len(both)-1})={t_stat:.2f}, p={p_val:.4f}, d={d_cohen:.2f}")
print(f"  iSD-RT: 1B M={isd_1b.mean():.1f}ms SD={isd_1b.std():.1f}ms, "
      f"2B M={isd_2b.mean():.1f}ms SD={isd_2b.std():.1f}ms, "
      f"t({len(both_isd)-1})={t_isd:.2f}, p={p_isd:.4f}, d={d_isd:.2f}")
print(f"  CPT CE: S1 M={ce1.mean():.3f} SD={ce1.std():.3f}, "
      f"S2 M={ce2.mean():.3f} SD={ce2.std():.3f}, retest r={r_ce:.3f}")
print(f"  CPTMEM face d': S1 M={dp1.mean():.3f} SD={dp1.std():.3f}, "
      f"S2 M={dp2.mean():.3f} SD={dp2.std():.3f}, retest r={r_dp:.3f}")
