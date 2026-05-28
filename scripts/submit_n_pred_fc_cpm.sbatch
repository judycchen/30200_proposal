#!/bin/bash
#SBATCH --job-name=n_pred_fc_cpm
#SBATCH --partition=caslake
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/n_pred_fc_cpm_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/n_pred_fc_cpm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase N (Aim 2): Compute FC from predicted task fMRI → LOO-CPM → behavior.
# Runs CPM for Nback (accuracy) and CPT2 (d-prime).
# 1000-permutation null distribution per task.
#
# Estimated runtime: ~30-60 minutes (LOO-CPM × 1000 perms × 2 tasks)
# Memory: 16G
#
# Prerequisites:
#   - l_apply_model_tasks.py complete: task _pred_SxRxT.mat files exist
#   - k_parse_timing_behavior.py complete: behavioral_scores.csv exists

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/n_cpm
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/n_cpm

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj
python scripts/n_pred_fc_cpm.py
