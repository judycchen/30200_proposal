#!/bin/bash
#SBATCH --job-name=c2_fnirs_controls
#SBATCH --partition=caslake
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/c2_fnirs_controls_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/c2_fnirs_controls_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase 3 supplement: Control analysis — Supp Table S2
# 3 controls × 1000 perms × 42 channels
# Estimated runtime: ~1-2 hours
# Prerequisites: c_fnirs_fmri_corr.py complete (fnirs_fmri_corr.mat exists)

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/c_fnirs-fmri-corr

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/c2_fnirs_fmri_corr_controls.py
