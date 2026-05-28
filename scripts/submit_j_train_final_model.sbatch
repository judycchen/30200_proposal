#!/bin/bash
#SBATCH --job-name=j_train_final
#SBATCH --partition=caslake
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/j_train_final_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/j_train_final_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase J: Train final aPCR model on all NNW fNIRS subjects and save weights.
#
# Estimated runtime: ~2 hours (updated for ~29 fNIRS subjects)
#   - Single training pass on ~29×59=1711 subject-pairs
#   - Training matrix ~1711×886 = ~1.5M rows (~38% larger than with 21 subjects)
#
# Memory: 128G — training matrix larger with expanded subject pool
# CPUs:   4   — sklearn PCA uses BLAS threading
#
# Prerequisites:
#   - b1/b2 complete: all_NNW_zhbo_TxRxS.mat updated with new subjects
#   - a_preproc-fmri.py complete: bold_TxRxS.npy exists

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/j_final-model

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj
python scripts/j_train_final_model.py
