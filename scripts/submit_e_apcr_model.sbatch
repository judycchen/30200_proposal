#!/bin/bash
#SBATCH --job-name=e_apcr_model
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/e_apcr_model_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/e_apcr_model_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Phase 4: aPCR predictive model — LOO cross-validation
#
# Estimated runtime: 6–10 hours (updated for ~29 fNIRS subjects)
#   - ~29 LOO iterations
#   - Each: concat 28×59=1652 pairs (1652×886=1.46M timepoints)
#   - PCA on ~1.46M×42 fNIRS and ~1.46M×122 fMRI matrices
#   - sklearn PCA is the bottleneck (~2× larger than with 21 subjects)
#
# Memory: 128G — training matrices ~2× larger than before
# CPUs: 4 — sklearn PCA uses multithreading via BLAS
#
# Prerequisites:
#   - b1/b2 preprocessing complete (5_excluded/ populated)
#   - a_preproc-fmri.py complete (bold_TxRxS.npy exists)
#   - helper scripts copied to scripts/helpers/
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/d_apcr-model

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/e_apcr_model.py
