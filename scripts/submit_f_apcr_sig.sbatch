#!/bin/bash
#SBATCH --job-name=f_apcr_sig
#SBATCH --partition=caslake
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/f_apcr_sig_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/f_apcr_sig_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 4: Permutation test + FDR correction + PNC
#
# Estimated runtime: 30–60 min
#   - 1000 permutations × 122 ROIs
#   - nltools phase_randomize per ROI
#   - PNC Cronbach's alpha over 59 fMRI subjects × 122 ROIs
#
# Memory: 32G — loads bold_TxRxS.npy (886×122×59 = 49M) into RAM
#
# Prerequisites:
#   - e_apcr_model.py complete (apcr_model_corr.mat and nnw_pred_RxT.mat exist)
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/f_apcr_significance.py
