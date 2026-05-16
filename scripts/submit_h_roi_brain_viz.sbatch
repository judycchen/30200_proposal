#!/bin/bash
#SBATCH --job-name=h_roi_brain_viz
#SBATCH --partition=caslake
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/h_roi_brain_viz_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/h_roi_brain_viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 5: ROI brain surface visualization
# Produces Figure 3a, Supp S1, Supp S2 using BrainNetViewer
# Estimated runtime: < 10 min
#
# WARNING: MATLAB license may fail on caslake compute nodes (Error -96)
# If so, run interactively on login node:
#   nohup matlab -nodisplay -nosplash -r \
#     "cd('scripts'); run('h_roi_brain_viz.m'); exit;" \
#     > logs/h_roi_brain_viz_manual.out 2>&1 &
#
# Prerequisites:
#   - f_apcr_significance.py complete (apcr_significance.mat exists)
#   - g_apcr_isfc.py complete (apcr_isfc.mat exists)
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/e_visualization/roi

module load matlab/2023a

cd /project/ycleong/users/judycchen/prediction-proj

matlab -nodisplay -nosplash -r \
    "cd('scripts'); run('h_roi_brain_viz.m'); exit;"
