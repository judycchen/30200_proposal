#!/bin/bash
#SBATCH --job-name=c_fnirs_fmri_corr
#SBATCH --partition=caslake
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/c_fnirs_fmri_corr_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/c_fnirs_fmri_corr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 3: fNIRS-fMRI one-to-one sanity check correlation
# Estimated runtime: ~20-40 min (1000 permutations x 42 channels)
# Memory: 16G sufficient — working with small group-mean arrays
#
# Prerequisites:
#   - b1_preproc_fnirs.m must be complete (5_excluded/ folder populated)
#   - a_preproc-fmri.py must be complete (bold_groupmean_TxR.npy exists)
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/c_fnirs_fmri_corr.py
