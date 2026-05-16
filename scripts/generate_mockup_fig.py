"""
generate_mockup_fig.py
Simulated scatter plot: Predicted DMN Activation vs. Reaction Time Variability
Expected effect size: r ≈ -0.40, based on Braun et al. (2015) and Finn & Bandettini (2021),
attenuated from fMRI-behavior r due to model-predicted rather than directly measured signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

np.random.seed(86)

N        = 21
target_r = -0.40

# Predicted DMN activation (z-scored)
dmn = np.random.normal(0, 1, N)

# Reaction time variability (SD of RTs in ms)
# Healthy young adults on n-back: ~150–260 ms SD, mean ~200 ms
rtv_noise = np.random.normal(0, 1, N)
rtv = 200 + 45 * (target_r * dmn + np.sqrt(1 - target_r**2) * rtv_noise)

r_val, p_val = stats.pearsonr(dmn, rtv)

fig, ax = plt.subplots(figsize=(5.5, 4.8))

# Scatter
ax.scatter(dmn, rtv, color='#2c5f8a', s=72, alpha=0.85, zorder=3,
           edgecolors='white', linewidths=0.6)

# Regression line
x_fit  = np.linspace(dmn.min() - 0.35, dmn.max() + 0.35, 300)
slope, intercept, *_ = stats.linregress(dmn, rtv)
ax.plot(x_fit, slope * x_fit + intercept, color='#c0392b', linewidth=1.8, zorder=2)

# Axis labels and title
ax.set_xlabel("Predicted DMN Mean Activation (z-score)", fontsize=11)
ax.set_ylabel("Reaction Time Variability — n-back (SD, ms)", fontsize=11)
ax.set_title("Predicted DMN Activation vs.\nReaction Time Variability",
             fontsize=12, fontweight='bold', pad=10)

# Annotations
p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < .001"
ax.text(0.97, 0.95, f"r = {r_val:.2f}, {p_str}",
        transform=ax.transAxes, fontsize=10, ha='right', va='top')
ax.text(0.97, 0.87, "Simulated data  |  N = 21",
        transform=ax.transAxes, fontsize=9,
        ha='right', va='top', color='#777777', style='italic')

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

plt.tight_layout()

out_dir  = "/project/ycleong/users/judycchen/prediction-proj/results/e_visualization/mockup"
out_path = os.path.join(out_dir, "fig_mockup_dmn_rtv.png")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
