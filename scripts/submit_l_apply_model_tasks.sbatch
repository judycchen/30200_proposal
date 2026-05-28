#!/bin/bash
#SBATCH --job-name=l_apply_tasks
#SBATCH --partition=caslake
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/l_apply_tasks_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/l_apply_tasks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase L: Apply final aPCR model to all task fNIRS data.
# Produces predicted fMRI time series for Nback, CPT1, CPT2, CPTMEM1, CPTMEM2.
#
# Estimated runtime: ~10-20 minutes (simple matrix multiplications, no training)
# Memory: 32G
#
# Prerequisites:
#   - j_train_final_model.py complete: apcr_final_model.npz exists
#   - b1/b2 complete: all task _zhbo_TxRxS.mat files exist in 5_excluded/

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/l_task-pred-fmri

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj
python scripts/l_apply_model_tasks.py
