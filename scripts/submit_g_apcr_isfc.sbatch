#!/bin/bash
#SBATCH --job-name=g_apcr_isfc
#SBATCH --partition=caslake
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/g_apcr_isfc_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/g_apcr_isfc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 4: ISFC validation of aPCR model
#
# Estimated runtime: 20–30 min
#   - LOO-ISFC on true fMRI (59 subjects × 122×122 matrix)
#   - LOO-ISFC on predicted fMRI (15 subjects × 122×122 matrix)
#   - 1000 permutation Mantel test (Pearson, ROI label shuffle)
#
# Memory: 32G — ISFC matrices are 59×122×122 and 15×122×122
#
# Prerequisites:
#   - e_apcr_model.py complete (nnw_pred_SxRxT.mat exists)
#   - Can run in parallel with f_apcr_significance.py
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/g_apcr_isfc.py
