#!/bin/bash
#SBATCH --job-name=d_mni_brain_viz
#SBATCH --partition=caslake
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/d_mni_brain_viz_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/d_mni_brain_viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 3 visualization: render fNIRS-fMRI r values on brain surface
# Estimated runtime: < 5 min
# Memory: 16G (SPM12 + BrainNetViewer can be memory hungry)
#
# Prerequisites:
#   - c_fnirs_fmri_corr.py must be complete (fnirs_fmri_corr.mat exists)
#
# WARNING: If MATLAB license fails on caslake (Error -96), run interactively
# on login node instead:
#   nohup matlab -nodisplay -nosplash -r \
#     "cd('scripts'); run('d_mni_brain_viz.m'); exit;" \
#     > logs/d_mni_brain_viz_manual.out 2>&1 &
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

module load matlab/2023a

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

matlab -nodisplay -nosplash -r \
    "cd('scripts'); run('d_mni_brain_viz.m'); exit;"
