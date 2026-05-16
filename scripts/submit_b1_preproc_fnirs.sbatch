#!/bin/bash
#SBATCH --job-name=b1_fnirs_preproc
#SBATCH --partition=caslake
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/b1_fnirs_preproc_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/b1_fnirs_preproc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# ─── NOTES ────────────────────────────────────────────────────────────────────
# Runs b1_preproc-fnirs.m (Sections 1–6):
#   load raw → extract NNW segment → AnalyzIR pipeline → QA → resample → ISC
#
# Estimated runtime: 1–2 hours for 21 subjects
#   - Section 2 (load + trigger extraction): ~5–10 min
#   - Section 3 (AnalyzIR pipeline):         ~45–90 min (wavelet filter is slow)
#   - Sections 4–6 (QA, resample, ISC):      ~10 min
#
# Memory: 32G — AnalyzIR loads full raw data objects into memory per subject
#
# After completion:
#   1. Check .out log for the ISC table printed at the end
#   2. Update exclS_NNW in b2_exclude-fnirs.m
#   3. Submit submit_b2_exclude-fnirs.sbatch
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs

module load matlab/2023a

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

matlab -nodisplay -nosplash -r "cd('scripts'); run('b1_preproc_fnirs.m'); exit;"
