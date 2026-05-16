#!/bin/bash
#SBATCH --job-name=i_viz_network_isfc
#SBATCH --partition=caslake
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ycleong/users/judycchen/prediction-proj/logs/i_viz_network_isfc_%j.out
#SBATCH --error=/project/ycleong/users/judycchen/prediction-proj/logs/i_viz_network_isfc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=judycchen@uchicago.edu
#SBATCH --account=pi-ycleong

# Phase 5: Figure 3b (network barplot) + Figure 4 (ISFC heatmaps)
# Estimated runtime: < 5 min
# Prerequisites: f_apcr_significance.py and g_apcr_isfc.py complete

mkdir -p /project/ycleong/users/judycchen/prediction-proj/logs
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/e_visualization/network
mkdir -p /project/ycleong/users/judycchen/prediction-proj/results/e_visualization/isfc

source /project/ycleong/users/judycchen/miniconda3/etc/profile.d/conda.sh
conda activate judycenv

cd /project/ycleong/users/judycchen/prediction-proj

python scripts/i_viz_network_isfc.py
