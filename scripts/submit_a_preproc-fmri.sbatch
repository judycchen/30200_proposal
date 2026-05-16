#!/bin/bash
#SBATCH --job-name=a_preproc-fmri
#SBATCH --partition=caslake
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/a_preproc-fmri_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/a_preproc-fmri_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Estimated runtime: 15–30 min for 59 subjects (bottleneck is disk I/O).
# 1 hour limit gives comfortable headroom.
# 16G memory sufficient — subjects loaded one at a time and discarded.
# ──────────────────────────────────────────────────────────────────────────────

# Create log directory if it doesn't exist
mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

# Load python module and activate environment
module load python/anaconda-2022.05
source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

# Run the script
cd /project/ycleong/users/judycchen/prediction-proj
python scripts/a_preproc-fmri.py
