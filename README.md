# Predicting fNIRS from fMRI: aPCR Model Pipeline

Analysis code for a project predicting fNIRS signals from fMRI data during naturalistic movie-watching, using an adaptive Principal Component Regression (aPCR) model. All analyses are run on the University of Chicago Research Computing Center's Midway3 HPC cluster.

## Repository Structure

```
prediction-proj/
├── scripts/               # Analysis pipeline scripts
├── data/                  # Intermediate and output data
│   ├── a_fmri-roi-ts/     # fMRI ROI time series
│   ├── b_fnirs-preproc/   # Preprocessed fNIRS data (not tracked; see Data Availability)
│   ├── c_fnirs-fmri-corr/ # fNIRS–fMRI spatial correlation results
│   ├── d_apcr-model/      # aPCR model outputs and predictions
│   └── ...                # Frontal and PCA75 variants
├── results/               # Figures and summary outputs
└── logs/                  # Slurm job logs
```

## Pipeline

Each step corresponds to a numbered script and a matching `.sbatch` file for Slurm submission:

| Step | Script | Description |
|------|--------|-------------|
| a | `a_preproc-fmri.py` | fMRI ROI time series extraction |
| b1 | `b1_preproc_fnirs.m` | fNIRS preprocessing (NIRS AnalyzIR Toolbox) |
| b2 | `b2_exclude_fnirs.m` | fNIRS channel/subject exclusion |
| c | `c_fnirs_fmri_corr.py` | fNIRS–fMRI spatial correlation |
| c2 | `c2_fnirs_fmri_corr_controls.py` | Control analyses for fNIRS–fMRI correlation |
| c3 | `c3_fnirs_fmri_corr_heatmap.py` | Correlation heatmap visualization |
| d | `d_mni_brain_viz.m` | MNI brain visualization |
| e | `e_apcr_model.py` | aPCR model training and prediction |
| f | `f_apcr_significance.py` | Permutation-based significance testing |
| g | `g_apcr_isfc.py` | Inter-Subject Functional Connectivity (ISFC) analysis |
| h | `h_roi_brain_viz.m` | ROI brain visualization |
| i | `i_viz_network_isfc.py/.m` | Network ISFC visualization |

## Dependencies

**Python:** NumPy, scikit-learn, nibabel, nilearn, BrainIAK

**MATLAB:** [NIRS AnalyzIR Toolbox](https://github.com/huppertt/nirs-toolbox) (included in `scripts/helpers/`)

## Data Availability

The fNIRS dataset was collected under an approved IRB protocol at the University of Chicago and will be made available upon reasonable request following study completion, in accordance with institutional data sharing policies.

The fMRI dataset (N = 59, NNW movie-watching) was obtained from a collaborating lab and is available through the original data sharing agreement.

Preprocessed behavioral data (trial-level reaction times and accuracy) will be shared alongside the analysis code upon publication.

The preprocessed fNIRS data (`data/b_fnirs-preproc/`) is excluded from this repository due to file size. It is stored on Midway3 at:

```
/project/ycleong/users/judycchen/prediction-proj/data/b_fnirs-preproc/
```
