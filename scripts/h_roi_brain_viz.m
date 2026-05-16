%% h_roi_brain_viz.m
% Phase 5: Convert per-ROI stats to NIfTI volumes and render onto brain surface
%
% Produces (equivalent to Gao et al. 2025):
%   Figure 3a  : grp-grp r values, FDR-thresholded, cortical surface
%   Supp Fig S1: PNC values, FDR-thresholded, cortical surface
%   Supp Fig S2: ISC — true fMRI ISC and predicted fMRI ISC maps
%
% Pipeline:
%   1. Load stats from apcr_significance.mat and apcr_isfc.mat
%   2. Paint each parcel in the atlas with its stat value → NIfTI
%   3. Render NIfTI onto brain surface using BrainNetViewer
%
% Dependencies (all in scripts/helpers/):
%   - NIFTI_tools/ (load_nii, save_nii)
%   - BrainNetViewer_20191031/
%   - figure3_config.mat
%
% Inputs:
%   - data/d_apcr-model/apcr_significance.mat
%   - data/d_apcr-model/apcr_isfc.mat
%   - data/a_fmri-roi-ts/atlas_122_resampled.nii.gz
%
% Outputs (results/e_visualization/roi/):
%   - fig3a_grp_grp_r_pmasked.nii.gz + .jpg
%   - suppS1_pnc_pmasked.nii.gz      + .jpg
%   - suppS2_true_isc.nii.gz         + .jpg
%   - suppS2_pred_isc.nii.gz         + .jpg

clear all; clc;

%% ── variant flags ────────────────────────────────────────────────────────────
% Must match the settings used in e_apcr_model.py / f_apcr_significance.py.
% Set PCA75 = true to visualize the 75%-variance PCA run (d_apcr-model_pca75).
% Set FRONTAL_ONLY = true to visualize the frontal-channels-only run.
% Both flags can be combined.

FRONTAL_ONLY = false;
PCA75        = true;

%% ── paths ────────────────────────────────────────────────────────────────────

BASE     = "/project/ycleong/users/judycchen/prediction-proj";
HELPERS  = fullfile(BASE, "scripts/helpers");

out_suffix = "";
if FRONTAL_ONLY; out_suffix = strcat(out_suffix, "_frontal"); end
if PCA75;        out_suffix = strcat(out_suffix, "_pca75");   end

OUT_DIR   = fullfile(BASE, strcat("results/e_visualization", out_suffix, "/roi"));
MODEL_DIR = fullfile(BASE, strcat("data/d_apcr-model", out_suffix));

if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end

addpath(genpath(fullfile(HELPERS, "NIFTI_tools")));
addpath(genpath(fullfile(HELPERS, "BrainNetViewer_20191031")));

ATLAS_FILE  = fullfile(BASE, "data/a_fmri-roi-ts/atlas_122_resampled.nii.gz");
CONFIG_FILE = fullfile(HELPERS, "figure3_config.mat");
SURF_FILE   = "BrainMesh_ICBM152_smoothed.nv";
SIG_FILE    = fullfile(MODEL_DIR, "apcr_significance.mat");
ISFC_FILE   = fullfile(MODEL_DIR, "apcr_isfc.mat");

%% ── load atlas ───────────────────────────────────────────────────────────────

fprintf("Loading atlas...\n");
atlas     = load_nii(char(ATLAS_FILE));
atlas_img = int16(atlas.img);
n_rois    = 122;
fprintf("Atlas shape: %d x %d x %d\n", size(atlas_img,1), size(atlas_img,2), size(atlas_img,3));

%% ── load significance results ────────────────────────────────────────────────

fprintf("Loading significance results...\n");
sig = load(SIG_FILE);

grp_grp_r = double(squeeze(sig.grp_grp_corr_Rx1));
sigmask   = logical(squeeze(sig.grp_grp_corr_sigmask_Rx1));
pnc       = double(squeeze(sig.pnc_Rx1));

fprintf("Significant ROIs: %d/122\n", sum(sigmask));

% FDR-masked: non-significant → NaN (renders transparent)
grp_grp_r_pmasked = grp_grp_r;
grp_grp_r_pmasked(~sigmask) = 0;   % BrainNetViewer uses 0 for masking

pnc_pmasked = pnc;
pnc_pmasked(~sigmask) = 0;
pnc_pmasked(isnan(pnc_pmasked)) = 0;

%% ── Figure 3a: grp-grp r, FDR-thresholded ───────────────────────────────────

fprintf("\n--- Figure 3a: grp-grp r (FDR-thresholded) ---\n");
render_roi_figure(grp_grp_r_pmasked, atlas, atlas_img, OUT_DIR, ...
                  "fig3a_grp_grp_r_pmasked", SURF_FILE, CONFIG_FILE);

%% ── Supp Fig S1: PNC, FDR-thresholded ───────────────────────────────────────

fprintf("\n--- Supp Fig S1: PNC (FDR-thresholded) ---\n");
render_roi_figure(pnc_pmasked, atlas, atlas_img, OUT_DIR, ...
                  "suppS1_pnc_pmasked", SURF_FILE, CONFIG_FILE);

%% ── Supp Fig S2: ISC maps ────────────────────────────────────────────────────

fprintf("\n--- Supp Fig S2: ISC maps ---\n");
isfc = load(ISFC_FILE);
true_isfc_RxR = double(isfc.true_isfc_RxR);
pred_isfc_RxR = double(isfc.pred_isfc_RxR);

true_isc_Rx1 = diag(true_isfc_RxR);
pred_isc_Rx1 = diag(pred_isfc_RxR);

fprintf("True ISC median: %.4f\n", median(true_isc_Rx1));
fprintf("Pred ISC median: %.4f\n", median(pred_isc_Rx1));

render_roi_figure(true_isc_Rx1, atlas, atlas_img, OUT_DIR, ...
                  "suppS2_true_isc", SURF_FILE, CONFIG_FILE);

render_roi_figure(pred_isc_Rx1, atlas, atlas_img, OUT_DIR, ...
                  "suppS2_pred_isc", SURF_FILE, CONFIG_FILE);

fprintf("\nDone: h_roi_brain_viz.m\n");
fprintf("All outputs in: %s\n", OUT_DIR);

%% ═══════════════════════════════════════════════════════════════════════════
%  HELPER FUNCTIONS — must be at end of script file
%% ═══════════════════════════════════════════════════════════════════════════

function nii_out = parcel_num2nii(data_Rx1, atlas, atlas_img)
    % Paint each ROI parcel with its scalar value
    out_map = zeros(size(atlas_img), 'double');
    for roi = 1:length(data_Rx1)
        val = data_Rx1(roi);
        if ~isnan(val)
            out_map(atlas_img == roi) = val;
        end
    end
    nii_out                   = atlas;
    nii_out.img               = out_map;
    nii_out.hdr.dime.datatype = 64;
end

function render_roi_figure(data_Rx1, atlas, atlas_img, out_dir, filename, surf_file, config_file)
    % Save NIfTI, gzip, and render to JPG via BrainNetViewer
    nii_path = char(fullfile(out_dir, strcat(filename, ".nii")));
    jpg_path = char(fullfile(out_dir, strcat(filename, ".jpg")));
    gz_path  = char(strcat(nii_path, ".gz"));

    % paint parcels and save
    nii_out = parcel_num2nii(data_Rx1, atlas, atlas_img);
    save_nii(nii_out, nii_path);
    fprintf("  NIfTI saved: %s\n", nii_path);

    % compress and clean up .nii
    gzip(nii_path);
    delete(nii_path);
    fprintf("  Compressed: %s\n", gz_path);

    % render surface figure
    BrainNet_MapCfg(char(surf_file), gz_path, char(config_file), jpg_path);
    fprintf("  Figure saved: %s\n", jpg_path);
end
