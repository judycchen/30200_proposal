#!/bin/bash
#SBATCH --job-name=k_parse_timing
#SBATCH --partition=caslake
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/k_parse_timing_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/k_parse_timing_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase K: Parse .tri trigger files and PsychoPy CSVs.
# Extracts block timing and behavioral scores for all subjects and tasks.
#
# Estimated runtime: < 5 minutes (pure file I/O, no heavy computation)
# Memory: 8G
#
# Prerequisites:
#   - Raw data present in /project/ycleong/datasets/CogTasks-NNW_judy/
#   - PsychoPy CSVs present in psychopy_data/nback/ and psychopy_data/cpt_dual/

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/data/k_task-timing

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj
python scripts/k_parse_timing_behavior.py
