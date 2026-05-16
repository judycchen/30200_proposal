#!/bin/bash
#SBATCH --job-name=b2_fnirs_exclude
#SBATCH --partition=caslake
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/b2_fnirs_exclude_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/b2_fnirs_exclude_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Runs b2_exclude-fnirs.m (Sections 7–8):
#   manual subject exclusion → post-exclusion ISC
#
# *** BEFORE SUBMITTING: update exclS_NNW in b2_exclude-fnirs.m ***
#
# Estimated runtime: < 5 minutes
# Memory: 8G — only loads the already-processed TxRxS matrix
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

module load matlab/2023a

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

matlab -nodisplay -nosplash -r "cd('scripts'); run('b2_exclude_fnirs.m'); exit;"

