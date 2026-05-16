#!/bin/bash
#SBATCH --job-name=e_apcr_model
#SBATCH --partition=caslake
#SBATCH --time=06:00:00
#SBATCH --mem=64G
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
# Estimated runtime: 2–4 hours
#   - 15 LOO iterations
#   - Each: concat 14×59=826 pairs (826×886=732k timepoints)
#   - PCA on ~732k×42 fNIRS and ~732k×122 fMRI matrices
#   - sklearn PCA is the bottleneck
#
# Memory: 64G — training matrices are large (826×886 = ~730k rows)
# CPUs: 4 — sklearn PCA uses multithreading via BLAS
#
# Prerequisites:
#   - b1/b2 preprocessing complete (5_excluded/ populated, 15 subjects)
#   - a_preproc-fmri.py complete (bold_TxRxS.npy exists)
#   - helper scripts copied to scripts/helpers/
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/d_apcr-model

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/e_apcr_model.py
