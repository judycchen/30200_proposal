#!/bin/bash
#SBATCH --job-name=m_task_glm
#SBATCH --partition=caslake
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/m_task_glm_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/m_task_glm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase M (Aim 1): GLM contrast on predicted N-back fMRI — 2-back vs 1-back.
# Outputs FDR-thresholded t-map on brain surface.
#
# Estimated runtime: ~15 minutes (GLM is fast; BrainNetViewer render is slow)
# Memory: 16G
#
# Prerequisites:
#   - l_apply_model_tasks.py complete: Nback_pred_SxRxT.mat exists
#   - k_parse_timing_behavior.py complete: nback_block_timing.mat exists
#   - MATLAB accessible for BrainNetViewer surface rendering (or Python NIfTI only)

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/m_task-glm
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/m_task-glm

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj
python scripts/m_task_glm_contrast.py
